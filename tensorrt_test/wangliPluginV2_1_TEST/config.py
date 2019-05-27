import graphsurgeon as gs
#my_soft_max = gs.create_plugin_node(name='my_soft_max_name', op='my_soft_max')
my_soft_max = gs.create_plugin_node(name='output_y', op='my_soft_max')
namespace_plugin_map = {'output_y': my_soft_max}
def preprocess(dynamic_graph):
    #print 'dynamic_graph : ', dir(dynamic_graph)
    #print 'dynamic_graph node map : ', dynamic_graph.node_map
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
