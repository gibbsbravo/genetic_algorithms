import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.model_selection import ParameterGrid
import imageio

#%% Define OpenCV helper functions

def show_image(img):
    """Creates a new window showing the image"""
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resizes the input without distortion 
        # Function source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def load_target_image(input_path='images/charmander.png', gray_thresh=150, target_width=16):
    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, gray_thresh, 255, 0)
    
    img = resize_image(img, width=target_width)
    _, img = cv2.threshold(img, 240, 255, 0)
    
    return img

def show_target_image(input_img, display_width=500):
    resized_img = resize_image(input_img, width=display_width, height=None, inter=cv2.INTER_NEAREST)
    show_image(resized_img)


#%% Define Parent Selection class
    
class ParentSelection(object):
    """Class which handles the different parent selection methods"""
    def __init__(self, selection_strategy='prop_selection', selection_strategy_params=None):
        self.selection_strategy = selection_strategy
        self.available_strategies = ['prop_selection', 'perf_cutoff', 'tournament_selection']
        self.selection_strategy_params = selection_strategy_params
    
    def get_parents_idx(self, fitness_scores):
        if self.selection_strategy == 'prop_selection':
            return self.prop_selection(fitness_scores)
        
        elif self.selection_strategy == 'perf_cutoff':
            return self.performance_cutoff(fitness_scores, self.selection_strategy_params['top_prop_selected'])
        
        elif self.selection_strategy == 'tournament_selection':
            return self.tournament_selection(fitness_scores, self.selection_strategy_params['n_candidates'])
        
        else:
            print('Please select valid crossover strategy')
    
    def prop_selection(self, fitness_scores):
        parent_a_idx, parent_b_idx = np.random.choice(
                a=np.arange(len(fitness_scores)), size=2, p=fitness_scores / fitness_scores.sum(), replace=False)
        
        return parent_a_idx, parent_b_idx
    
    def performance_cutoff(self, fitness_scores, top_prop_selected):
        top_n = int(len(fitness_scores) * top_prop_selected)
        top_idx = np.argpartition(fitness_scores, -top_n)[-top_n:]
        parent_a_idx, parent_b_idx = np.random.choice(a=top_idx, size=2, replace=False)
        
        return parent_a_idx, parent_b_idx
    
    def tournament_selection(self, fitness_scores, n_candidates):
        candidates = np.random.choice(a=np.arange(len(fitness_scores)), size=(2, n_candidates), replace=False)
        parent_a_idx = candidates[0, np.argmax(fitness_scores[candidates[0]])]
        parent_b_idx = candidates[1, np.argmax(fitness_scores[candidates[1]])]
        
        return parent_a_idx, parent_b_idx
    
#%% Define Crossover class

class Crossover(object):
    """Class which handles the different crossover methods"""
    def __init__(self, crossover_strategy='uniform', crossover_params=None):
        self.crossover_strategy = crossover_strategy
        self.available_stragies = ['uniform', 'single_point', 'multi_point']
        
        # Dict containing optional args if specific crossover strategy is selected
        self.crossover_params = crossover_params
        
    def crossover(self, parent_a, parent_b):
        if self.crossover_strategy == 'uniform':
            return self.uniform_crossover(parent_a, parent_b)
        
        elif self.crossover_strategy == 'single_point':
            return self.single_point_crossover(parent_a, parent_b, prop_a=self.crossover_params['prop_a'])
        
        elif self.crossover_strategy == 'multi_point':
            return self.multi_point_crossover(
                    parent_a, parent_b, n_crossover_points=self.crossover_params['n_crossover_points'])
        else:
            print('Please select valid crossover strategy')
    
    def uniform_crossover(self, parent_a, parent_b):
        """Uses uniform crossover"""
        assert len(parent_a) == len(parent_b), 'Lengths are not equal'
        
        idx = np.random.choice([True, False], len(parent_a))
        
        return np.where(idx, parent_a, parent_b)
    
    def single_point_crossover(self, parent_a, parent_b, prop_a=None):
        """A crossover point on both chromosomes is picked randomly and they are swapped
        Can also optionally be set as proportion a"""
        assert len(parent_a) == len(parent_b), 'Lengths are not equal'
        
        if prop_a == None:
            # set crossover point randomly
            idx = np.random.randint(0, len(parent_a))
            
        else:
            idx = int(prop_a * len(parent_a))
        
        return np.concatenate((parent_a[:idx], parent_b[idx:]))
    
    def multi_point_crossover(self, parent_a, parent_b, n_crossover_points=2):
        """Set multiple crossover points and cross between them"""
        idx = np.sort(np.random.randint(1, len(parent_a)-1, n_crossover_points))
        idx = np.concatenate([[0], idx, [len(parent_a)]])
        
        sequence = []
        
        for cross_idx, (previous, current) in enumerate(zip(idx, idx[1:])):
            if cross_idx+1 % 2 == 0:
                sequence.extend(parent_a[previous:current])
            else:
                sequence.extend(parent_b[previous:current])
    
        return np.array(sequence)

#%% Define Population class 

class Population(object):
    """Class which contains the current population and defines the step function 
        which advances the population one generation"""
    def __init__(self, population_size, crossover_method, selection_method,
                 n_offspring, mutation_rate, correct_img):
        self.population_size = population_size
        self.crossover_method = crossover_method
        self.selection_method = selection_method
        self.n_offspring = n_offspring
        self.mutation_rate = mutation_rate
        self.correct_img = correct_img
        self.correct_chromosome = correct_img.flatten().reshape(1,-1)
        self.length = self.correct_chromosome.shape[1]
        # Initialize the population
        self.parents = np.random.randint(2, size=(self.population_size, self.length), dtype='uint8') * 255
        
        # Tracking
        self.mean_scores = []
        self.max_scores = []
        self.best_parent = []
        self.n_generations = 0
        
    def get_fitness_scores(self):
        return (self.parents == self.correct_chromosome).sum(1) / self.length
    
    def get_children(self, fitness_scores):
        children = []
    
        for _ in range(int(self.population_size / 2)):
            # Select parents using selection method
            parent_a, parent_b = self.parents[np.array(self.selection_method.get_parents_idx(fitness_scores))]
            
            # Create children
            children.extend(np.array([self.crossover_method.crossover(parent_a, parent_b) for _ in range(self.n_offspring)]))
        
        children = np.array(children, dtype='uint8')
        
        if len(children) > self.population_size:
            # Randomly select children up to the population limit
            children = children[np.random.choice(len(children), size=self.population_size, replace=False)]
        
        return children
    
    def mutate(self):
        mutation_idx = np.random.random(self.parents.shape) < self.mutation_rate
        
        # Flip bits on randomly mutated cells
        self.parents[mutation_idx] = 255-self.parents[mutation_idx]
        
    def step(self):
        """Apply one iteration of evolution"""
        self.n_generations += 1
        
        # Calculate the fitness of current parents and append results for output
        fitness_scores = self.get_fitness_scores()
        self.mean_scores.append(fitness_scores.mean())
        self.max_scores.append(fitness_scores.max())
        self.best_parent.append(self.parents[np.argmax(fitness_scores)])
        
        # Create new children based on best parents
        self.parents = self.get_children(fitness_scores)
        
        # Mutate population
        self.mutate()
        
    def show_evolution(self, delay=100, display_width=500, gif_export_name=None):
        """Displays a video of the evolution"""
        frames = []
        
        for gen_number, (img, max_score) in enumerate(zip(self.best_parent, self.max_scores)):
            resized_img = resize_image(img.reshape(self.correct_img.shape),
                                       width=display_width, height=None, inter=cv2.INTER_NEAREST)
            
            header = np.ones((50, resized_img.shape[1]), dtype=np.uint8) * 255
            resized_img = np.vstack([header, resized_img])
            cv2.putText(resized_img,'Generation: ' + str(gen_number+1) + "    " + 'Fitness Score: ' + str(np.around(max_score, 2)),
                        (int(display_width/20), 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2, 0)
            
            if gif_export_name is not None:
                frames.append(resized_img)
            
            cv2.imshow('image', resized_img)
            cv2.waitKey(delay)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if gif_export_name is not None:
            imageio.mimsave(gif_export_name, frames, duration=delay/1000)
        
    def plot_performance(self):
        """Plots the performance of the fitness by generation"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.mean_scores)
        plt.title("Performance by Generation")
        plt.xlabel("Generation Number")
        plt.ylabel("Fitness Score")
        plt.show()
            
    def reset(self):
        self.parents = np.random.randint(2, size=(self.population_size, self.length)) * 255
        
        # Tracking
        self.mean_scores = []
        self.max_scores = []
        self.best_parent = []
        self.n_generations = 0

#%% Define GA function 

def apply_genetic_algorithm(correct_img, population_size=100, n_offspring=2, mutation_rate=0.00,
                      selection_strategy='prop_selection', selection_strategy_params=None,
                      crossover_strategy='uniform', crossover_params=None,
                      max_n_iters=500, final_accuracy_thresh=0.99, verbose=False, print_every=1):    
    # Initialize the crossover & selection methods and the initial population
    selection_method = ParentSelection(selection_strategy=selection_strategy, 
                                       selection_strategy_params=selection_strategy_params)
    crossover_method = Crossover(crossover_strategy=crossover_strategy, crossover_params=crossover_params)
    
    
    population = Population(population_size=population_size,
                            crossover_method=crossover_method,
                            selection_method=selection_method,
                            n_offspring=n_offspring,
                            mutation_rate=mutation_rate, 
                            correct_img=correct_img)
    
    start_time = time.time()
    
    for gen_number in range(max_n_iters):
        # Advance one generation
        population.step()
        
        if verbose and gen_number % print_every == 0:
            print("Iteration: {} | Mean Score: {:.4f} | Max Score: {:.4f}".format(
                    gen_number+1, population.mean_scores[-1], population.max_scores[-1]))
            
        if (population.max_scores[-1] >= final_accuracy_thresh) or \
            ((len(population.max_scores) >= 10) and (np.ptp(population.max_scores[-10:-1]) <= 0.001)):
            if verbose:
                print("Iteration: {} | Mean Score: {:.4f} | Max Score: {:.4f}".format(
                        gen_number+1, population.mean_scores[-1], population.max_scores[-1]))
            break
   
    elapsed_time = time.time() - start_time
    
    return population, gen_number+1, population.max_scores[-1], elapsed_time

#%% Gridsearch defined hyperparameters 

# Set random seed 
np.random.seed(34)

correct_img = load_target_image(input_path='images/charmander_180.png', gray_thresh=180, target_width=64)
#show_target_image(correct_img)

selection_strategy_params = {'top_prop_selected' : 0.25, 'n_candidates' : 5}
crossover_params = {'prop_a' : None, 'n_crossover_points' : 2}

param_grid = {'population_size': [50, 100, 500, 1000],
              'mutation_rate' : [0.00, 0.001, 0.01],
              'selection_strategy': ['perf_cutoff', 'tournament_selection'],
              'crossover_strategy' : ['uniform', 'single_point']
              }


def main():
    grid = ParameterGrid(param_grid)
    number_of_configs = 48
    gridsearch_params = np.random.choice(list(grid), replace=False, size=number_of_configs)
    
    results = np.zeros((number_of_configs, 3))
    
    for idx, gs_params in enumerate(gridsearch_params):
        converged_population, final_gen_number, max_score, elapsed_time = apply_genetic_algorithm(
                correct_img=correct_img, 
                population_size=gs_params['population_size'],
                n_offspring=2, 
                mutation_rate=gs_params['mutation_rate'],
                selection_strategy=gs_params['selection_strategy'], 
                selection_strategy_params=selection_strategy_params,
                crossover_strategy=gs_params['crossover_strategy'], 
                crossover_params=crossover_params,
                max_n_iters=500, final_accuracy_thresh=0.99, verbose=False, print_every=1)
        
        results[idx, :] = final_gen_number, max_score, elapsed_time
        
        print()
        print('Completed config: ', idx+1, gs_params,
              'final_gen_number: ', final_gen_number,
              'max_score: ', max_score,
              'elapsed_time: ', elapsed_time)
    
    results_df = pd.concat([pd.DataFrame(list(gridsearch_params)), 
                          pd.DataFrame(results, columns=['final_gen_number', 'max_score', 'elapsed_time'])], axis=1)
    
    return results_df

# Export results
#results_df.to_csv('charmander_gridsearch_results.csv', index=False)
    
#%% Run single file

correct_img = load_target_image(input_path='images/charmander_180.png', gray_thresh=180, target_width=64)
show_target_image(correct_img)
population_size = 1000
n_offspring = 2
mutation_rate = 0.001

selection_strategy = 'perf_cutoff' # Available strategies include: ['prop_selection', 'perf_cutoff', 'tournament_selection']
selection_strategy_params = {'top_prop_selected' : 0.25, 'n_candidates' : 5}
crossover_strategy = 'uniform' # Available strategies include: ['uniform', 'single_point', 'multi_point']
crossover_params = {'prop_a' : None, 'n_crossover_points' : 2}

max_n_iters = 1000
final_accuracy_thresh = 0.99
verbose = True
print_every = 1

converged_population, final_gen_number, max_score, elapsed_time =  apply_genetic_algorithm(
            correct_img=correct_img, 
            population_size=population_size,
            n_offspring=2, 
            mutation_rate=mutation_rate,
            selection_strategy=selection_strategy, 
            selection_strategy_params=selection_strategy_params,
            crossover_strategy=crossover_strategy, 
            crossover_params=crossover_params,
            max_n_iters=max_n_iters, final_accuracy_thresh=final_accuracy_thresh, 
            verbose=verbose, print_every=print_every)

#%% Plot evolution

# Show video of evoluation
converged_population.show_evolution(delay=40, display_width=700, gif_export_name=None)

# Plot performance
#converged_population.plot_performance()
