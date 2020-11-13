import torch
import math

def readfile(iter_start, iter_end, file):
    file.seek(iter_start)
    while (iter_start < iter_end):
        line = file.readline()
        yield line
        iter_start = file.tell()

class MyIterableDataset(torch.utils.data.IterableDataset):
     def __init__(self, start, end , filename):
         super(MyIterableDataset).__init__()
         assert end > start, "this example code only works with end >= start"
         self.start = start
         self.end = end
         self.filename = filename

     def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)

         fd = open(self.filename, 'r')
         if iter_start != 0:
             fd.seek(iter_start - 1)
             if fd.read(1) != '\n':
                 line = fd.readline()
                 iter_start = fd.tell()

         fd.seek(iter_start)
         while (iter_start < iter_end):
             line = fd.readline()
             line_list = line.strip().split(",")

             yield torch.tensor(list(map(float,line_list)))
             iter_start = fd.tell()
         fd.close()



filename = "t3"
file = open(filename,'r')
begin = file.tell()
file.seek(0, 2)
end = file.tell()
print("begin=",begin)
print("end=",end)
file.close()
ds = MyIterableDataset(start=begin, end=end, filename=filename)

ele = torch.utils.data.DataLoader(ds, num_workers=2, pin_memory=True)
for e in ele:
    print(e)
