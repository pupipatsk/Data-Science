# Monte carlo approximation of pi
# Approach: 1/4Circle in unit Square

import random
import math
import time

start_time = time.time()

total_count = 0
inside_count = 0

iterations = 1000000
for i in range(iterations):
    total_count += 1
    
    x = random.random() # Returns a random float number between 0 and 1
    y = random.random()

    # distance = math.sqrt(x**2 + y**2)
    distance2 = x**2 + y**2
    
    if distance2 <= 1:
        inside_count += 1

print(f'Pi is approximately: {4 * inside_count / total_count}')
end_time = time.time()
print(f'Iterations: {iterations:,} | Time: {end_time - start_time :.2f} seconds')