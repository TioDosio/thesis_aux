import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

rospy.init_node('send_goal_node')

client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
client.wait_for_server()

goal = MoveBaseGoal()
goal.target_pose.header.frame_id = "map"
goal.target_pose.header.stamp = rospy.Time.now()
goal.target_pose.pose.position.x = 10   # Set your desired x
goal.target_pose.pose.position.y = 5  # Set your desired y
goal.target_pose.pose.orientation.w = 0.5  # Facing forward

client.send_goal(goal)
client.wait_for_result()

if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
    rospy.loginfo("Goal reached!")
else:
    rospy.loginfo("Failed to reach goal.")
