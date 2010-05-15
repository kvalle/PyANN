import copy

# Parse options dict by augmenting with defaults dict
def parse_options(options, defaults, final = {}):
    for key in defaults:
        if isinstance(defaults[key], dict):
            final[key] = {}
            if not key in options: options[key] = {} 
            parse_options(options[key], defaults[key], final[key])
            continue
        final[key] = options[key] if key in options else defaults[key]
    for key in options:
        if key not in defaults: final[key] = options[key] 
    result = copy.copy(final)
    final = {}
    return result

def string_to_exact_len(s, n):
    s = str.strip(str(s))
    l = len(s)
    if l < n:
        return s + ' ' * (n-l)
    elif n < l:
        end = (n-l)
        return s[0:end]
    else:
        return s
        