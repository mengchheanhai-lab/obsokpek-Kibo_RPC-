#!/usr/bin/python3
# -*- coding:utf-8 -*-

#this is a test for github
#chhean needs another test
"""
Kibo-RPC (7th) preliminary-round solution, built on the official template.

Strategy
  1. Read all CheckPoints from /mission/get_checkpoints.
  2. Compute the shortest visiting order (exact Held-Karp DP, fine for <= 13 points),
     starting at the Dock Station and ending at the goal point if one is identifiable.
  3. Fly to each CheckPoint (position + attitude), then report it so the 1-second
     dwell / AR-marker judgment can happen.
  4. Watch the 5-minute budget: if time is running short, skip the remaining
     CheckPoints and go straight to the goal / finish report (the time multiplier
     and Pt_goal are only awarded if the report is made inside the limit).
"""
import json
import math
import time
import rospy
import actionlib
from platform_msgs.msg import UserLogic
from platform_msgs.srv import StopProcessingUserNode, StopProcessingUserNodeResponse
from geometry_msgs.msg import PoseStamped
from krpc_ib2_msgs.srv import MissionStart, MissionFinished, CheckPoints
from krpc_ib2_msgs.msg import CheckpointPassedAction, CheckpointPassedGoal
from ib2_msgs.msg import CtlCommandAction, CtlCommandGoal, CtlCommandResult, CtlStatusType

# ----------------------------- tunables ---------------------------------
TIME_LIMIT = 300.0          # 5 minutes (rulebook 2.2.2 #6)
SAFETY_MARGIN = 25.0        # seconds kept in reserve for goal move + final report
MOVE_RETRIES = 2            # attempts per CheckPoint move
DOCK_POSITION = (0.200, 0.0, 0.0)   # Table 2.2-3 start position
# ------------------------------------------------------------------------


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def normalize_quat(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / n for c in q]


def plan_order(start, points, goal=None):
    """Shortest path start -> all `points` (list of xyz) -> optional `goal`.
    Returns list of indices into `points`. Exact Held-Karp DP."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]
    INF = float('inf')
    d = [[dist(points[i], points[j]) for j in range(n)] for i in range(n)]
    d0 = [dist(start, points[i]) for i in range(n)]
    dg = [dist(points[i], goal) for i in range(n)] if goal is not None else [0.0] * n

    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = d0[i]
    for mask in range(1, 1 << n):
        for last in range(n):
            cur = dp[mask][last]
            if cur == INF or not (mask >> last) & 1:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                c = cur + d[last][nxt]
                if c < dp[nm][nxt]:
                    dp[nm][nxt] = c
                    parent[nm][nxt] = last
    best_last = min(range(n), key=lambda i: dp[full][i] + dg[i])
    order, mask, cur = [], full, best_last
    while cur != -1:
        order.append(cur)
        prev = parent[mask][cur]
        mask ^= (1 << cur)
        cur = prev
    order.reverse()
    return order


class UserSolution:

    def __init__(self):
        rospy.init_node('krpc_user')

        # Action client
        self.__command = actionlib.SimpleActionClient('/ctl/command', CtlCommandAction)
        self.__checkpoint_client = actionlib.SimpleActionClient('/event/checkpoint_passed', CheckpointPassedAction)

        # Service client
        self.__mission_start = rospy.ServiceProxy('/event/mission_start', MissionStart)
        self.__mission_finished = rospy.ServiceProxy('/event/mission_finished', MissionFinished)
        self.__checkpoints = rospy.ServiceProxy('/mission/get_checkpoints', CheckPoints)

        # Subscriber
        self.__subscriber_start = rospy.Subscriber('/ib2_user/start', UserLogic, self.__callback_start)  # DO NOT REMOVE THIS LINE

        # Service server
        self.__stop_processing_server = rospy.Service('/ib2_user/stop', StopProcessingUserNode, self.__stop_processing)  # DO NOT REMOVE THIS LINE

        self.__started = False
        self.__action_active = False
        self.__stop_requested = False
        self.__t0 = None
        rospy.set_param('/ib2_user/ready', True)  # DO NOT REMOVE THIS LINE

    # ------------------------------------------------------------------
    # time helpers
    # ------------------------------------------------------------------
    def __time_left(self):
        if self.__t0 is None:
            return TIME_LIMIT
        return TIME_LIMIT - (time.time() - self.__t0)

    def __should_abort(self):
        return rospy.is_shutdown() or self.__stop_requested

    # ------------------------------------------------------------------
    # mission notifications (same as template)
    # ------------------------------------------------------------------
    def __notify_mission_start(self):
        rospy.wait_for_service('/event/mission_start')
        res = self.__mission_start()
        rospy.loginfo('mission_start result: success=%s message=%s', res.success, res.message)
        return res.success

    def __notify_mission_finished(self):
        rospy.wait_for_service('/event/mission_finished')
        res = self.__mission_finished()
        rospy.loginfo('mission_finished result: success=%s message=%s', res.success, res.message)
        return res.success

    def __notify_checkpoint_passed(self, checkpoint_id):
        rospy.loginfo('notify checkpoint_id=%d passed', checkpoint_id)
        self.__checkpoint_client.wait_for_server()
        goal = CheckpointPassedGoal()
        goal.checkpoint_id = checkpoint_id
        self.__checkpoint_client.send_goal(goal)
        while not self.__should_abort():
            if self.__checkpoint_client.wait_for_result(rospy.Duration(0.2)):
                result = self.__checkpoint_client.get_result()
                rospy.loginfo('checkpoint_passed result: %s', result)
                return result.success
        self.__checkpoint_client.cancel_goal()
        rospy.logwarn('checkpoint_passed action cancelled')
        return False

    # ------------------------------------------------------------------
    # checkpoint parsing
    # ------------------------------------------------------------------
    @staticmethod
    def __get_checkpoint_id(cp):
        for key in ('checkpoint_id', 'checkpointid', 'id'):
            if key in cp:
                return int(cp[key])
        return None

    @staticmethod
    def __is_goal(cp):
        for key in ('is_goal', 'goal', 'isGoal'):
            if cp.get(key) is True:
                return True
        t = str(cp.get('type', '')).lower()
        return 'goal' in t

    def __load_checkpoints(self):
        """Returns list of dicts: id, position[3], quaternion[4], is_goal."""
        rospy.wait_for_service('/mission/get_checkpoints')
        res = self.__checkpoints()
        rospy.loginfo('get_checkpoints result: success=%s message=%s', res.success, res.message)
        if not res.success:
            return []
        try:
            payload = json.loads(res.message)
            raw = payload.get('checkpoints', payload.get('Checkpoints', []))
        except Exception as e:
            rospy.logwarn('Failed to parse checkpoints payload: %s', e)
            return []

        result = []
        for cp in raw:
            if not isinstance(cp, dict):
                continue
            cid = self.__get_checkpoint_id(cp)
            pos = cp.get('position')
            if cid is None or not (isinstance(pos, list) and len(pos) >= 3):
                continue
            quat = [0.0, 0.0, 0.0, 1.0]
            att = cp.get('attitude')
            if isinstance(att, dict):
                q = att.get('quaternion')
                if isinstance(q, list) and len(q) >= 4:
                    quat = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            result.append({
                'checkpoint_id': cid,
                'position': [float(pos[0]), float(pos[1]), float(pos[2])],
                'quaternion': normalize_quat(quat),
                'is_goal': self.__is_goal(cp),
            })
        return result

    # ------------------------------------------------------------------
    # motion
    # ------------------------------------------------------------------
    def __move_once(self, target):
        position, quaternion = target['position'], target['quaternion']
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quaternion

        command_type = CtlStatusType(type=CtlStatusType.MOVE_TO_ABSOLUTE_TARGET)
        goal = CtlCommandGoal(target=pose, type=command_type)
        self.__command.wait_for_server()
        self.__action_active = True
        try:
            self.__command.send_goal(goal)
            while not self.__should_abort():
                if self.__command.wait_for_result(rospy.Duration(0.2)):
                    result = self.__command.get_result()
                    rospy.loginfo('move command result: %s', result)
                    return result.type == CtlCommandResult.TERMINATE_SUCCESS
                if self.__time_left() <= 0:
                    self.__command.cancel_goal()
                    return False
            self.__command.cancel_goal()
            return False
        finally:
            self.__action_active = False

    def __move_to_checkpoint(self, target):
        rospy.loginfo('moving to checkpoint_id=%d pos=%s quat=%s',
                      target['checkpoint_id'], target['position'], target['quaternion'])
        for attempt in range(1, MOVE_RETRIES + 1):
            if self.__should_abort() or self.__time_left() <= 0:
                return False
            if self.__move_once(target):
                return True
            rospy.logwarn('move to checkpoint %d failed (attempt %d/%d)',
                          target['checkpoint_id'], attempt, MOVE_RETRIES)
        return False

    # ------------------------------------------------------------------
    # main flow
    # ------------------------------------------------------------------
    def __run_user_flow(self, msg):
        try:
            if not self.__notify_mission_start():
                return
            self.__t0 = time.time()

            # USER EDIT AREA: define route planning and game-specific flow here.
            # ***********************************************
            cps = self.__load_checkpoints()
            goal_cps = [c for c in cps if c['is_goal']]
            goal_cp = goal_cps[0] if goal_cps else None
            targets = [c for c in cps if not c['is_goal']]

            order = plan_order(DOCK_POSITION,
                               [c['position'] for c in targets],
                               goal_cp['position'] if goal_cp else None)
            ordered = [targets[i] for i in order]
            rospy.loginfo('planned order (ids): %s', [c['checkpoint_id'] for c in ordered])

            current = list(DOCK_POSITION)
            speed = None  # m/s, learned from the first completed leg (includes dwell)

            for cp in ordered:
                if self.__should_abort():
                    break
                leg = dist(current, cp['position'])
                back = dist(cp['position'], goal_cp['position']) if goal_cp else 0.0
                need = SAFETY_MARGIN
                if speed:
                    need += (leg + back) / speed
                if self.__time_left() < need:
                    rospy.logwarn('time low (%.1fs left, need %.1fs) -> skip remaining checkpoints',
                                  self.__time_left(), need)
                    break

                t_leg = time.time()
                if self.__move_to_checkpoint(cp):
                    if self.__notify_checkpoint_passed(cp['checkpoint_id']):
                        dt = time.time() - t_leg
                        if leg > 0.05 and dt > 0.5:
                            s = leg / dt
                            speed = s if speed is None else 0.5 * (speed + s)
                current = cp['position']  # robot is (roughly) here either way

            # finish at the goal (reached with whatever time remains)
            if goal_cp and not self.__should_abort() and self.__time_left() > 0:
                if self.__move_to_checkpoint(goal_cp):
                    self.__notify_checkpoint_passed(goal_cp['checkpoint_id'])
            # ***********************************************

            if not rospy.is_shutdown():
                self.__notify_mission_finished()
        except Exception as exc:
            rospy.logerr('user flow failed: %s', exc)

    def __callback_start(self, msg):
        if self.__started:
            rospy.logwarn('start callback is already processed')
            return
        self.__started = True
        rospy.loginfo('start %s', msg)
        self.__run_user_flow(msg)

    def __stop_processing(self, request):
        try:
            self.__stop_requested = True
            if self.__action_active:
                self.__command.cancel_goal()
            rospy.loginfo('finish processing')
            return StopProcessingUserNodeResponse(result=StopProcessingUserNodeResponse.SUCCESS)
        except Exception as exc:
            rospy.logerr('stop processing failed: %s', exc)
            return StopProcessingUserNodeResponse(result=StopProcessingUserNodeResponse.ERROR)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    UserSolution().run()