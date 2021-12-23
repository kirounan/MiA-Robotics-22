#! /usr/bin/env python3
import rospy
from std_msgs.msg import float32

state = 0


def callback(data):
    uncert = 8.0
    Q = 3.0  # process variance
    R = 1.0  # sensor variance
    predstate = data.data
    preduncert = uncert + Q
    y = data.data - predstate #residual
    k = predstate / (predstate + R)
    state = predstate + k * y
    uncert = (1-k)*preduncert

def listner():
    rospy.Subscriber("OG", Float32, callback)
    rospy.spin()


def talker():
    pub = rospy.Publisher('chatter', Float32, queue_size=10)
    rospy.init_node('KalmanFilter', anonymous=True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.loginfo(state)
        pub.publish(state)
        rate.sleep()


if __name__ == '__main__':
    try:
        listner()
    except rospy.ROSInterruptException:
        pass
    talker()
