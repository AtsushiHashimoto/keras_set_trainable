import re

def make_pats(pat_strs, check_str_end=True):
    if check_str_end:
        for i,s in enumerate(pat_strs):
            if s[-2] == '\Z':
                continue
            pat_strs[i] = s + '\Z'
    return [re.compile(s) for s in pat_strs]

def in_re(s, pats):
    matchs = [p.match(s) for p in pats]
    matchs = [m.groups() for m in matchs if m is not None]
    return len(matchs)>0

'''
  model: a keras model who has layers in depth.
  trainable: True/False, which is set to layers
  targets: name pattern list to which the above 'trainable' value is set.
  check_str_end: control whether to match layer name by only head part correspondence or entire correspondence.
  e.g.) when check_str_end=False, targets = ['test']  matches to any layers whose name start with 'test'. 
  Namely, 'test01' and 'test02' are counted as targets.
  when check_str_end=True, targets = ['test'] matches only the layer whose name is 'test'. In this case,
  'test01' and 'test02' are excluded from the targets.
'''
def set_trainable(model,trainable, targets=None, check_str_end = True):
    is_target = True    
    set_objects = []
    if targets and len(targets)>0:
        if isinstance(targets[0], str):
            targets = make_pats(targets)
        is_target = (in_re(model.name,targets))
        
    #print("trainable of %s"%model.name,trainable) 
    if 'trainable' in model.get_config().keys():
        if is_target:
            set_objects.append(model.name)
            model.trainable = trainable
        
    if 'layers' in model.__dict__:
        for l in model.layers:
            #print("l.name: ",l.name)
            if is_target:
                set_objects += set_trainable(l,trainable)
            else:
                set_objects += set_trainable(l,trainable,targets=targets)
            
    elif 'layer' in model.__dict__:
        #print("layer: ",model.layer.trainable)
        if is_target:
            set_objects += set_trainable(model.layer,trainable)
        else:
            set_objects += set_trainable(model.layer,trainable,targets=targets)
            
    return set_objects
            
