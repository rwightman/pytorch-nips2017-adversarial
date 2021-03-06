import sys
import torch
import torch.utils.data
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
    mp.set_start_method('spawn')
except RuntimeError:
    pass


class MpFeeder:
    def __init__(self, dataset, maxsize=2):
        self.queue = mp.Queue(maxsize=maxsize)
        self.dataset = dataset
        self.is_done = mp.Event()
        self.is_shutdown = mp.Event()
        self.process = mp.Process(target=self._run)
        self.process.start()

    def _run(self):
        try:
            print("Entering run loop")
            sys.stdout.flush()
            while not self.is_shutdown.is_set():
                for i, element in enumerate(self.dataset):
                    torch.cuda.synchronize()
                    self.queue.put(element)
                    if self.is_shutdown.is_set():
                        break
                self.queue.put(None)

            print("Waiting on done")
            self.is_done.wait()
        except Exception as e:
            self.queue.put(e)
            self.is_shutdown.set()
            self.queue.close()

    def shutdown(self):
        self.is_shutdown.set()
        self.queue.close()

    def done(self):
        self.is_done.set()

    def __iter__(self):
        while True:
            output = self.queue.get()
            if output is None:
                break
            yield output

    def __len__(self):
        return len(self.dataset)


class TestDataset():
    def __init__(self):
        self.num = 999
        self.img_size = (3, 64, 64)

    def __getitem__(self, index):
        img = torch.rand(self.img_size).cuda()
        target = torch.LongTensor([index]).cuda()
        return img, target

    def __len__(self):
        return self.num


def test():
    num_iterations = 100000

    test_dataset = TestDataset()
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    q = MpFeeder(loader)
    for i in range(num_iterations):
        for qi, (input, target) in enumerate(q):
            print(qi, target)
        print("done, loop %d" % i)

    q.shutdown()
    q.done()


if __name__ == '__main__':
    test()


