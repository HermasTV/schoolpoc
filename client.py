'''demo of school counter
    @authors:   Eslam Abdelrahman
                Mohamed Salah
    @lisence: Tahaluf 2023
'''
# from school import school
from school_final import school
import time
from models import Models 
from concurrent.futures import ThreadPoolExecutor


models = Models()
pipe= school(models)
start= time.time()

with ThreadPoolExecutor(max_workers=4) as executor:
    print("starting pipeline")
    executor.map(pipe.run_1_stream,[0,1,2,3])
    print("Ending pipeline")
# run 4 streams in parallel using multiprocessing
print(f'pipline took: {(time.time()-start)*1000} ms')
