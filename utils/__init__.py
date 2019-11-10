import pickle, os, glob

def save_config(config, base_dir):
    tmp_device = config.device
    config.device = None
    pickle.dump(config, open(os.path.join(base_dir, 'config.dump'), 'wb'))
    config.device = tmp_device

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_directory(path):
    try:
        os.makedirs(path)
    except OSError:
        files = glob.glob(os.path.join(path, '*.*'))
        for f in files:
            os.remove(f)

# NOTE: modified from modular drl. add a link here
class LinearSchedule:
    def __init__(self, start, end=None, decay=None, max_steps=None):
        steps = None
        if end is None:
            end = start
            steps = 1
        else:
            steps = max_steps*decay

        self.inc = (start - end) / float(steps)
        self.start = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        return self.bound(self.start - self.inc*steps, self.end)

class PiecewiseSchedule:
    def __init__(self, start, end=None, decay=None, max_steps=None):
        assert(type(end) == list and type(decay) == list), '[Error] Expected lists of epsilon end and epsilon decay parameters'

        # determine the last tstep for each section
        self.section_ends = [0] + [d*max_steps for d in decay]
        
        #Find start and end eps for each piece
        bounds = [start] + end
        piece_bounds = [(bounds[i-1], bounds[i]) for i in range(1, len(bounds))]
        
        # calc total steps in each piece
        piece_steps = []
        total = 0
        for d in decay:
            tmp = d*max_steps
            fraction = tmp-total
            total += fraction
            piece_steps.append(fraction)

        #declare a linear schedule for each piece
        self.schedulers = []
        for idx, _ in enumerate(end):
            s, e = piece_bounds[idx]
            d = 1.0 #decay across whole piece
            max_step = piece_steps[idx]
            self.schedulers.append(LinearSchedule(s, e, d, max_step))

    def __call__(self, tstep=1):
        for idx, e in enumerate(self.section_ends):
            if tstep < e:
                return self.schedulers[idx-1](tstep-self.section_ends[idx-1])
        return self.schedulers[-1](tstep-self.section_ends[-2])

            