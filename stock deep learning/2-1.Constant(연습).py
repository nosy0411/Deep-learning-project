import tensorflow as tf

a = tf.add(2,3)
b = tf.multiply(5, a)

sess=tf.Session()
aa = sess.run(a)
bb= sess.run(b)

print(aa, bb)
write=tf.summary.FileWriter('./mygraph', sess.graph)

# 1부터 100까지 합을 계산한다
n = 100

# 1부터 100까지 합을 계산하는 그래프를 생성한다
g1 = tf.Graph()
with g1.as_default():
    mySum = tf.constant(0, name='mySum')
    for i in range(1, n + 1):
        mySum = tf.add(mySum, i)

# g1 그래프 영역의 그래프를 실행한다
with tf.Session(graph=g1) as sess:
    print("\n1부터 %d까지 합계 = %d" % (n, sess.run(mySum)))

# 결과 :
# 1부터 100까지 합계 = 5050

# g1에 생성된 그래프의 구조를 확인한다.
#print(g1.as_graph_def())
