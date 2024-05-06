from time import sleep

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

time = 0.05
n = 1011
n_pool = 10


def run(data):
    for i in tqdm(data):
        sleep(time)


def main():
    data = list(range(n))
    ls = len(data) // n_pool
    batches = [data[i : i + ls] for i in range(0, len(data), ls)]

    with ProcessPoolExecutor(max_workers=n_pool) as executor:
        executor.imap(run, batches)


if __name__ == "__main__":
    main()
