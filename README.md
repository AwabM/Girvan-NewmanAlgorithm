# Girvan-NewmanAlgorithm
This Repository give a python implementation of Girvan–Newman Algorithm that used to detect communities by fining betweenness among network's vertices. [Read more about the algorithm](https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm)


# Goal:
The code is divided into two parts to detect the communities in the network graph:
1- Betweenness Calculation: build a graph from the input file and find the Betweenness between vertices.
2- Community Detection: recursively finding new networks by cutting one edge and calculate modularity. Finally find the best cut among all the networks.
[Read more about Modularity equation](https://en.wikipedia.org/wiki/Modularity_(networks)) 

# Data:
Prepossessed version of Network Data Repository dataset (http://networkrepository.com). Each row in the file represent an edge in the network. The numbers represent IDs of vertices.
Total edges in this dataset are 906 edges in the dataset provided.
 power_input


# Files:
main_code.py: python code that get the network as an input and output two files (betweenness and community)
Betweenness.txt : first output of the code that has betweenness degree for every two vertices.
communities.txt: second output of the code that has communities achieved highest modularity.
data folder: contain power_input.txt that has the data described above.

# Excution:
spark-submit main_code.py <input_file_path> <betweenness_output_file_path> <community_output_file_path>
