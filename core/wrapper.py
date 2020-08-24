import functools
import inspect
from multiprocessing import Queue, Process


def processing_class(cls):
    class _WrappedClass:
        def __init__(self, obj):
            self._obj = obj
            self._send = Queue()
            self._recv = Queue()
            self._process: Process

            methods = inspect.getmembers(self._obj, inspect.ismethod)
            for name, _ in methods:
                if name[0] != '_':
                    f = lambda *args, **kwargs: self._dispatch(name, *args, **kwargs)
                    argspec = inspect.getfullargspec(getattr(obj, name))
                    sig_f = get_func_with_sig(f, name, argspec)
                    setattr(self, name, sig_f)

            def process_main(recv_queue, send_queue, obj):
                print('Process created!')
                while True:
                    data = recv_queue.get()
                    name = data['name']
                    args = data['args']
                    kwargs = data['kwargs']
                    result = getattr(obj, name)(*args, **kwargs)
                    send_queue.put(result)

            self._process = Process(target=process_main, args=(self._send, self._recv, self._obj))
            self._process.start()
            
        def _dispatch(self, name, *args, **kwargs):
            data = {
                'name': name,
                'args': args,
                'kwargs': kwargs,
            }
            self._send.put(data)
            return self._recv.get()
        
        def __del__(self):
            self._process.terminate()
        
    @functools.wraps(cls)
    def inner(*args, **kwargs) -> cls:
        obj = cls(*args, **kwargs)
        wrapped = _WrappedClass(obj)
        return wrapped
    return inner


def restore_arglist(argspec: inspect.FullArgSpec, allow_self):
    arglist = []
    call_arglist = []

    for i in argspec.args[:len(argspec.args)-len(argspec.defaults if argspec.defaults else [])]:
        if not allow_self and i != 'self':
            arglist.append(i)
            call_arglist.append(i)
    if argspec.defaults:
        arglist += [f'{a}="{d}"' if type(d)==str else f'{a}={d}' for a, d in zip(argspec.args[::-1], argspec.defaults[::-1])][::-1]
        call_arglist += [f'{a}="{d}"' if type(d)==str else f'{a}={d}' for a, d in zip(argspec.args[::-1], argspec.defaults[::-1])][::-1]
    if argspec.varargs:
        arglist.append(f'*{argspec.varargs}')
        call_arglist.append(f'*{argspec.varargs}')
    elif len(argspec.kwonlyargs) > 0:
        arglist.append('*')
        call_arglist.append('*')
    for i in argspec.kwonlyargs[:len(argspec.kwonlyargs)-len(argspec.kwonlydefaults if argspec.kwonlydefaults else [])]:
        arglist.append(i)
    for i in argspec.kwonlyargs:
        call_arglist.append(f'{i}={i}')
    if argspec.kwonlydefaults:
        arglist += [f'{a}="{d}"' if type(d)==str else f'{a}={d}' for a, d in zip(argspec.kwonlyargs[::-1], argspec.kwonlydefaults[::-1])][::-1]
    if argspec.varkw:
        arglist.append(f'**{argspec.varkw}')
        call_arglist.append(f'**{argspec.varkw}')
    return arglist, call_arglist


def get_func_with_sig(f, name, argspec, allow_self=False):
    arglist, call_arglist = restore_arglist(argspec, allow_self)
    argstr = ", ".join(arglist)
    call_argstr = ", ".join(call_arglist)
    fakefunc = f'def {name}({argstr}):\n    return real_func({call_argstr})\n'
    fakefunc_code = compile(fakefunc, "fakesource", "exec")
    fakeglobals = {}
    eval(fakefunc_code, {"real_func": f}, fakeglobals)
    f_with_good_sig = fakeglobals[name]
    return f_with_good_sig
