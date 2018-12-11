# CS685-project
Development of a genetic algorithm to calculate accuracy-energy consumption tradeoff for a on-body sensor device


GA.py is the main library with the function I use in the other two files (multiple_run_plot.py, plot_iterations.py).

To execute multiple_run_plot.py, make sure to pass the population size as an argument (suggested values are 8, 16, 32)


- multiple_run_plot.py performs the actual algorithm with the values set on the library. It prints the best population when the threshold is met by the fitness score together with the actual score and the amount of iterations needed. If the threshold is not met, it still prints the same values and explicitly says that convergence was not met. The genetic algorithm runs 10 times

- plot_iterations.py plots a graph of the amount of iterations that have been needed for 3 different population sizes (8, 16 and 32). For each population size, 10 values are plotted because they represent the 10 executions performed.
