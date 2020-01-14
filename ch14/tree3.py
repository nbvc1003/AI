from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import io
from sklearn.tree import DecisionTreeClassifier
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


data = load_iris()
y = data.target
X = data.data[:,2:] # 꽃잎의 길이와 폭

feature_names = data.feature_names[2:]

# 앤트로피가 작을수로 규칙이 있다.
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1,
                                  random_state=0) # 앤트로피가 작은것을 우선으로
tree1.fit(X,y)
def draw_decision_tree(model, name):
    dot_buf = io.StringIO() # 이미지 파일 버퍼

    # 다이어그램을 이미미 파일로 출력
    export_graphviz(model, out_file=dot_buf, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png() # 이미지 객체..
    f = open(name, "wb")
    f.write(image)
    f.close()

    return Image(image)
draw_decision_tree(tree1, 'tree1.png')


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, tree1.predict(X)))


tree2 = DecisionTreeClassifier(criterion='entropy', max_depth=2,
                                  random_state=0) # 앤트로피가 작은것을 우선으로
tree2.fit(X,y)
draw_decision_tree(tree2, 'tree2.png')
print(confusion_matrix(y, tree2.predict(X)))

tree3 = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                                  random_state=0).fit(X,y) # 앤트로피가 작은것을 우선으로
draw_decision_tree(tree3, 'tree3.png')
print(confusion_matrix(y, tree3.predict(X)))
