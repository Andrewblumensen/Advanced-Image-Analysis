import imageio
import numpy as np
import matplotlib.pyplot as plt
import simple_snake as sis

filename = 'C:/Users/hujo8/OneDrive/Gamle kurser/Advanced image analysis/6 Week/crawling_amoeba.mov'
vid = imageio.get_reader(filename)
movie = np.array([im for im in vid.iter_data()], dtype=np.float)/255
movie = np.mean(movie, axis=3)

#%% settings
nr_points = 100
step_size = 10
alpha = 0.1
beta = 0.1
center = np.array([120,200])
radius = 40

#%% initialization
snake = sis.make_circular_snake(nr_points, center, radius)
B = sis.regularization_matrix(nr_points, alpha, beta)
frame = movie[0]
fig, ax = plt.subplots()
ax.imshow(frame, cmap='gray')
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'b-')

for i in range(50):    
    fig, ax = plt.subplots()
    snake = sis.evolve_snake(snake, frame, B, step_size)    
    ax.clear()
    ax.imshow(frame, cmap='gray')
    ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'b-')
    ax.set_title(f'initialization, iter {i}')
    
  
      
#%% tracking
for i in range(0,477): 
    fig, ax = plt.subplots()
    frame = movie[i] 
    snake = sis.evolve_snake(snake, frame, B, step_size)    
    ax.clear()
    ax.imshow(frame, cmap='gray')
    ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'b-')
    ax.set_title(f'tracking, frame {i}')
    





