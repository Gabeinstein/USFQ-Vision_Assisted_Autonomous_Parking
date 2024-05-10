#!/usr/bin/env python3
# encoding: utf-8

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty
from std_msgs.msg import Bool

moveBindings = {
    'w': (1, 0),
    'e': (1, -1),
    'a': (0, 1),
    'd': (0, -1),
    'q': (1, 1),
    'x': (-1, 0),
    'c': (-1, 1),
    'z': (-1, -1),
    'W': (1, 0),
    'E': (1, -1),
    'A': (0, 1),
    'D': (0, -1),
    'Q': (1, 1),
    'Z': (-1, -1),
}

speedBindings = {
    '1': (1.1, 1.1),
    '2': (.9, .9),
    '3': (1.1, 1),
    '4': (.9, 1),
    '5': (1, 1.1),
    '6': (1, .9),
}

parkingBindings = {
    'p': (1),
    'P': (1),
}

parking = False

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist: key = sys.stdin.read(1)
    else: key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)
def callback_parking(msg):
    global parking

    if (msg.data == True):
        parking = True
    else:
        parking = False

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('transbot_keyboard')
    sub = rospy.Subscriber('/parking/enable', Bool, callback_parking)
    linear_limit = rospy.get_param('~linear_limit', 0.45)
    angular_limit = rospy.get_param('~angular_limit', 2.0)
    pub = rospy.Publisher('~/cmd_vel', Twist, queue_size=1)
    pub_parking_enable = rospy.Publisher('/parking/enable',Bool,queue_size=1)
    (speed, turn) = (0.2, 1.0)
    (x, th) = (0, 0)
    status = 0
    count = 0
    try:
        print(vels(speed, turn))
        
        while (1):
            if parking == False:
                key = getKey()
                if key in moveBindings.keys():
                    x = moveBindings[key][0]
                    th = moveBindings[key][1]
                    count = 0
                elif key in speedBindings.keys():
                    speed = speed * speedBindings[key][0]
                    turn = turn * speedBindings[key][1]
                    count = 0
                    if speed > linear_limit: speed = linear_limit
                    if turn > angular_limit: turn = angular_limit
                    print(vels(speed, turn))
                    status = (status + 1) % 15
                elif key in parkingBindings.keys():
                    enable_parking = Bool()
                    enable_parking.data = True
                    pub_parking_enable.publish(enable_parking)
                
                elif key == ' ': 
                    (x, th) = (0, 0)
                    enable_parking = Bool()
                    enable_parking.data = False
                    pub_parking_enable.publish(enable_parking)
                else:
                    count = count + 1
                    if count > 4: (x, th) = (0, 0)
                    if (key == '\x03'): break
  
                twist = Twist()
                twist.linear.x = speed * x
                twist.angular.z = turn * th
                pub.publish(twist)
    except Exception as e: print(e)
    finally: pub.publish(Twist())
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)