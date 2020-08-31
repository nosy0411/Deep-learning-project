import tensorflow as tf

# Ex (1) : Varible에 1 부터 100 까지의 합이 누적된다.
# =================================================

# 1부터 100까지 합을 계산한다
n = 100
tf.reset_default_graph()
mySum = tf.Variable(0, name='mySum')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1, n + 1):
    sess.run(tf.assign(mySum, tf.add(mySum, i)))
    
print("\n1부터 %d까지 합계 = %d" % (n, sess.run(mySum)))
sess.close()
    
g = tf.get_default_graph()
g.as_graph_def()

# 결과 :
# 1부터 100까지 합계 = 5050

# Ex (2) : For 문으로 생성된 Assign Operation이 전부 실행되지 못하고, 
#          마지막 Assign 만 수행된다. 잘못된 결과임.
# =================================================================

# 1부터 100까지 합을 계산한다
n = 100
tf.reset_default_graph()
mySum = tf.Variable(0, name='mySum')
for i in range(1, n + 1):
    myAdd = tf.add(mySum, i)
    myUpdate = tf.assign(mySum, myAdd)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(myUpdate)
    
print("\n1부터 %d까지 합계 = %d" % (n, sess.run(mySum)))
sess.close()
    
# 결과 :
# 1부터 100까지 합계 = 100


# Ex (3) : 원하는 결과가 나오기는 하지만, Variable에 누적 결과가 저장되는 
#          것이 아니라 mySum이 Tensor로 바뀐다. mySum이 Variable이 아님.
# =====================================================================

# 1부터 100까지 합을 계산한다
n = 100
tf.reset_default_graph()
mySum = tf.Variable(0, name='mySum')
print("\nBefore : mySum --> ", mySum)
for i in range(1, n + 1):
    mySum = tf.add(mySum, i)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(mySum)
    
print("1부터 %d까지 합계 = %d" % (n, sess.run(mySum)))
print("After : mySum --> ", mySum)
sess.close()
    
# 결과 :
# Before : mySum -->  <tf.Variable 'mySum:0' shape=() dtype=int32_ref>
# 1부터 100까지 합계 = 5050
# After : mySum -->  Tensor("Add_99:0", shape=(), dtype=int32)
