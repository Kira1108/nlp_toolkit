from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import time
from .database import TextDB

TextCallable = Callable[[str], BaseModel]

@dataclass
class MultiThreadTask:
    parser:TextCallable
    db:TextDB
    batch_sleep:float = 0.2
    max_size:int = 10
    verbose:bool = True

    def parse_single(self, id:int, text:str):
        try:
            result = self.parser(text)
            if self.verbose:
                print(result)
            self.db.update_by_id(id,parse_result = result.json())
        except Exception as e:
            print("Parse error\n" + e)
            pass

    def parse(self, offset:int = 0, size:int = 10):
        if size > self.max_size:
            print(f"Concurrent size {size} is too large. Consider using a smaller size.")

        current_batch = self.db.get_unparsed_limit_offset(limit =size, offset = offset)

        if len(current_batch) == 0:
            print("No more unprased text.")
            return False

        results = []

        with ThreadPoolExecutor(max_workers=size) as executor:
            for text in current_batch:
                future = executor.submit(self.parse_single, text.id, text =text.text)
                results.append(future)

        time.sleep(self.batch_sleep)
        print("Finished Batch.")
        return True

    def __call__(self, n_batches:int = 2, offset:int = 0, size:int = 10):

        i = 0
        while True:
            has_any = self.parse(offset = offset, size = size)
            if not has_any:
                print("Finished")
                break
            i+=1
            if i >= n_batches:
                print("Stopped")
                break