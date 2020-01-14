import pydot
command = """
digraph G {Hello->World}
"""
graph = pydot.graph_from_dot_data(command)[0]
image = graph.create_png()
from IPython.core.display import Image
Image(image)

f = open("test.png", "wb")
f.write(image)
f.close()

command = """
digraph G {
main -> parse -> execute;
main -> init;
main -> cleanup;
execute -> make_string;
execute -> printf
init -> make_string;
main -> printf;
execute -> compare;
}
"""
graph = pydot.graph_from_dot_data(command)[0]
image = graph.create_png()
Image(image)

f = open("test2.png", "wb")
f.write(image)
f.close()




