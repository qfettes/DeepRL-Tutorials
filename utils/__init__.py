import pickle, os, glob, math

def save_config(config, base_dir, valid_arguments):
    print(f'{" Hyperparameters " :*^50}')
    for var, value in vars(config).items():
        if var in valid_arguments:
            print(f'{colorize(f"{var}", color="blue", bold=True) :-<36}' + f'{f"{value}" :->25}')
    print(f'{" End Hyperparameters " :*^50}')
    pickle.dump(config, open(os.path.join(base_dir, 'config.dump'), 'wb'))

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
        assert(len(end) == len(decay)), '[Error] Expected equal number of decay lengths and targets for epsilon'

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

class ExponentialSchedule:
    def __init__(self, start, end, decay, max_steps):
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, tstep):
        return self.end + (self.start - self.end) * math.exp(-1. * tstep / self.decay)


# Borrowed from OpenAI SpinningUp
#   https://github.com/openai/spinningup
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

# Borrowed from OpenAI SpinningUp
#   https://github.com/openai/spinningup
def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)      