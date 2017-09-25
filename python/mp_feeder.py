import sys
import torch
import torch.utils.data
import torch.multiprocessing as mp
#import multiprocessing as mp
try:
    #mp.set_sharing_strategy('file_system')
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
                for i, (input, true_target, adv_target) in enumerate(self.dataset):
                    self.queue.put((input, true_target, adv_target))
                    if self.is_shutdown.is_set():
                        break
                self.queue.put((None, None, None))

                print("Waiting on done")
                self.is_done.wait()
        except Exception as e:
            print("Ahhhh", str(e))
            self.queue.put((Exception(), None, None))
            self.is_shutdown.set()
            self.queue.close()

    def shutdown(self):
        self.is_shutdown.set()
        self.queue.close()

    def done(self):
        self.is_done.set()

    def __iter__(self):
        while True:
            input, true_target, adv_target = self.queue.get()
            if input is None:
                break
            yield input, true_target, adv_target

    def __len__(self):
        return len(self.dataset)


class TestDataset():
    def __init__(self):
        self.num = 999
        self.img_size = (3, 64, 64)

    def __getitem__(self, index):
        img = torch.rand(self.img_size).cuda()
        true_target = torch.LongTensor([index]).cuda()
        adv_target = torch.LongTensor([index]).cuda()
        return img, true_target, adv_target

    def __len__(self):
        return self.num


def test():
    test_dataset = TestDataset()

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    q = MpFeeder(loader)

    count = 0
    while True:
        for i, (input, true_target, adv_target) in enumerate(q):
            print(i, true_target)
        count += 1
        print("done, loop %d" % count)

    q.shutdown()

    q.done()


if __name__ == '__main__':
    test()


