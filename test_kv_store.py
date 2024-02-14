from kv_store import *

import subprocess as sp

def get_gpu_memory(message:str=''):
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    print(f'@{message}: cuda memory used: {memory_use_values}')
    # return memory_use_values

cuda.cuInit(0)
get_gpu_memory('init')
id = create_kv(0,200000000000,1)
get_gpu_memory('after create kv')
addr_1 = increase_kv_size(id, 300)
get_gpu_memory('after first increase 300')
torch_tensor = ptr_to_tensor(addr_1, 300, (300,))
addr_2 = increase_kv_size(id, 600)
get_gpu_memory('after second increase 600')
torch_tensor_2 = ptr_to_tensor(addr_2, 600, (600,))
torch_tensor_3 = ptr_to_tensor(addr_1, 900, (900,))


def print_func(message:str=''):
    print(f'---{message}: \n torch_tensor ptr: {torch_tensor.data_ptr()} \n, value: {torch_tensor} ')
    print(f'---{message}: \ntorch_tensor_2 ptr: {torch_tensor_2.data_ptr()} \n, value: {torch_tensor_2} ')
    print(f'---{message}: \ntorch_tensor_3 ptr: {torch_tensor_3.data_ptr()} \n, value: {torch_tensor_3} ')

torch_tensor[:] = 1
print_func('assign 1 to torch_tensor')
torch_tensor_2[:] = 2
print_func('assign 2 to torch_tensor_2')

get_gpu_memory('after assign values')

destory_kv(id)

get_gpu_memory('after destory kv')

del torch_tensor_3, torch_tensor_2, torch_tensor

get_gpu_memory('after del tensors')

destory_kv(id)

get_gpu_memory('destroy kv after del tensors')

print('done')
