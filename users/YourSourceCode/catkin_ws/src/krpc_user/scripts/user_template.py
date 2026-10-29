#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Kibo-RPC (7th) preliminary solution v6.

v6 (from the real v4 run log):
  * End-game rule: skipping ANY checkpoint forfeits the goal bonus (needs all checkpoints), so
    when 4 or fewer remain and finishing them all is predicted to fit (small margin), go for it.
    v4 skipped the last checkpoint with 73 s left although ~60 s were enough.
  * Speed scale is the average real/predicted ratio over all legs (less jumpy than the last leg).
  * If the pose error just before a report is large (e.g. 0.08 m / 4.6 deg gave only Lv2), do one
    corrective move first.  Navigation frame was verified against the log (errors ~0.01 m).
  * Logs when the robot gets close vs when the controller finishes (to study early reporting).

v5 notes (v4 + calmer route planning).

v5: rotation weight lowered (the 8-leg fit could not support a strong rotation term, which
    produced zig-zag routes); ROUTE_MODE switch: "optimal" (default) or "sweep".

Kibo-RPC v4 notes:

What the real v3 run (8 checkpoints) showed, and what v4 does about it:
  * v3 skipped CP7 with 93 s left (its safety check was far too pessimistic). A skipped
    checkpoint costs its own points AND the goal bonus (Pt_goal needs ALL checkpoints),
    and it left 61 s of the 300 s unused.
  * Leg time is well described by   t = A + B*metres + C*radians_of_rotation
    (fitted on the real legs, error ~4 s).  v4 plans the visiting order by predicted
    TIME (distance + rotation), not distance only.
  * v4 re-plans after every leg from the robot's real position, learns a speed scale from
    how long legs really take, and only drops checkpoints (the most expensive detours
    first) when the remaining time truly cannot cover them plus the goal leg.
  * The goal checkpoint (payload "goalpoint_id") is always last.
  * All timing uses the ROS/sim clock; the completion report is always sent before the
    5-minute limit (late reports score nothing, rulebook 2.2.3).
  * Logs the pose error just before each report (diagnostics only, does not change flight).
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

try:  # optional: only used for diagnostics
    from ib2_msgs.msg import Navigation
except ImportError:  # pragma: no cover
    Navigation = None

# ----------------------------- tunables ---------------------------------
TIME_LIMIT = 300.0           # 5 minutes (rulebook 2.2.2 #6)
SAFETY_MARGIN = 8.0          # sim-seconds of reserve kept after the planned goal leg
FINISH_MARGIN = 4.0          # end-game: margin when going for "all remaining checkpoints + goal"
ENDGAME_COUNT = 4            # end-game rule applies when this many (or fewer) checkpoints remain
CORRECT_POS = 0.05           # m   - corrective move if the pose error before reporting exceeds this
CORRECT_ATT = 4.0            # deg - ... or this
HARD_FLOOR = 3.0             # below this many seconds left, stop moving and just report
PLAN_PAD = 1.05              # planning pessimism on top of the learned scale
HOLD_SECONDS = 1.5           # settle time after arrival, before reporting (needs >= 1 s)
MOVE_RETRIES = 2             # attempts to complete one move command
NOTIFY_RETRIES = 3           # move+hold+report cycles per checkpoint
DOCK_POSITION = (0.200, 0.0, 0.0)        # Table 2.2-3 start position
DOCK_QUATERNION = (0.0, 0.0, 0.0, 1.0)   # Table 2.2-3 start attitude

# Leg time model (sim seconds, includes the settle + report): A + B*metres + C*radians
TIME_A, TIME_B, TIME_C = 18.4, 7.5, 1.5

# Route mode:
#   "optimal": cheapest order by predicted time (distance + a little rotation). Flies past some
#              checkpoints and comes back for them when that is shorter overall.
#   "sweep"  : visit checkpoints in order of distance from the dock (nearest first, out to the
#              far end), then fly straight to the goal.
ROUTE_MODE = "optimal"

# Optional: fixed goal pose (x, y, z, qx, qy, qz, qw). Used only if the payload has no goal.
GOAL_POSE = None
# ------------------------------------------------------------------------


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def normalize_quat(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / n for c in q]


def quat_angle(q1, q2):
    """Rotation angle (rad) between two attitudes; q and -q are the same attitude."""
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def leg_seconds(a, b):
    """Predicted time for one leg. a, b = (position, quaternion)."""
    return TIME_A + TIME_B * dist(a[0], b[0]) + TIME_C * quat_angle(a[1], b[1])


def _vec3(v):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return [float(v[0]), float(v[1]), float(v[2])]
    if isinstance(v, dict) and all(k in v for k in ('x', 'y', 'z')):
        return [float(v['x']), float(v['y']), float(v['z'])]
    return None


def _quat4(v):
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
    if isinstance(v, dict) and all(k in v for k in ('x', 'y', 'z', 'w')):
        return [float(v['x']), float(v['y']), float(v['z']), float(v['w'])]
    return None


def find_position(cp):
    for c in (cp.get('position'),
              cp['pose'].get('position') if isinstance(cp.get('pose'), dict) else None):
        p = _vec3(c)
        if p:
            return p
    return None


def find_quaternion(cp):
    cands = []
    att = cp.get('attitude')
    if isinstance(att, dict):
        cands += [att.get('quaternion'), att.get('orientation')]
    cands += [att, cp.get('quaternion'), cp.get('orientation')]
    if isinstance(cp.get('pose'), dict):
        cands.append(cp['pose'].get('orientation'))
    for c in cands:
        q = _quat4(c)
        if q:
            return normalize_quat(q)
    return None


def node_of(cp):
    return (cp['position'], cp['quaternion'])


def plan_route(start, nodes, goal=None):
    """Cheapest order (by predicted time) start -> all `nodes` -> optional `goal`.
    start/nodes/goal are (position, quaternion). Exact Held-Karp DP.
    Returns (list of indices into nodes, predicted seconds)."""
    n = len(nodes)
    if n == 0:
        return [], (leg_seconds(start, goal) if goal else 0.0)
    INF = float('inf')
    c = [[leg_seconds(nodes[i], nodes[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    c0 = [leg_seconds(start, nodes[i]) for i in range(n)]
    cg = [leg_seconds(nodes[i], goal) for i in range(n)] if goal else [0.0] * n

    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = c0[i]
    for mask in range(1, 1 << n):
        row = dp[mask]
        for last in range(n):
            cur = row[last]
            if cur == INF:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                v = cur + c[last][nxt]
                if v < dp[nm][nxt]:
                    dp[nm][nxt] = v
                    parent[nm][nxt] = last
    best_last = min(range(n), key=lambda i: dp[full][i] + cg[i])
    total = dp[full][best_last] + cg[best_last]
    order, mask, cur = [], full, best_last
    while cur != -1:
        order.append(cur)
        prev = parent[mask][cur]
        mask ^= (1 << cur)
        cur = prev
    order.reverse()
    return order, total


def plan_sweep(start, nodes, goal=None):
    """Nearest-to-the-dock first (ascending x), then the goal. Returns (order, predicted seconds)."""
    order = sorted(range(len(nodes)), key=lambda i: nodes[i][0][0])
    seq = [start] + [nodes[i] for i in order] + ([goal] if goal else [])
    return order, sum(leg_seconds(seq[k], seq[k + 1]) for k in range(len(seq) - 1))


def select_route(start, cps, goal, budget):
    """Best visiting order for `cps` whose predicted time (incl. the goal leg) fits `budget`.
    If it does not fit, drop the checkpoint whose removal saves the most time, repeat.
    Returns (ordered list of cps, predicted seconds)."""
    items = list(cps)
    while True:
        planner = plan_sweep if ROUTE_MODE == "sweep" else plan_route
        order, total = planner(start, [node_of(c) for c in items], goal)
        route = [items[i] for i in order]
        if total <= budget or not route:
            return route, total
        seq = [start] + [node_of(c) for c in route] + ([goal] if goal else [])
        best = None
        for k in range(1, len(route) + 1):
            prev, cur = seq[k - 1], seq[k]
            nxt = seq[k + 1] if k + 1 < len(seq) else None
            save = leg_seconds(prev, cur)
            if nxt:
                save += leg_seconds(cur, nxt) - leg_seconds(prev, nxt)
            if best is None or save > best[0]:
                best = (save, k - 1)
        victim = route[best[1]]
        items = [c for c in items if c is not victim]


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
        self.__nav = None
        self.__nav_time = -1e9
        if Navigation is not None:
            self.__subscriber_nav = rospy.Subscriber('/sensor_fusion/navigation', Navigation, self.__callback_nav)

        # Service server
        self.__stop_processing_server = rospy.Service('/ib2_user/stop', StopProcessingUserNode, self.__stop_processing)  # DO NOT REMOVE THIS LINE

        self.__started = False
        self.__action_active = False
        self.__stop_requested = False
        self.__t0 = None
        self.__wall_clock = False
        rospy.set_param('/ib2_user/ready', True)  # DO NOT REMOVE THIS LINE

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def __now(self):
        """Seconds on the ROS clock (sim time in the simulator); wall time as a fallback."""
        if self.__wall_clock:
            return time.time()
        return rospy.get_time()

    def __time_left(self):
        if self.__t0 is None:
            return TIME_LIMIT
        return TIME_LIMIT - (self.__now() - self.__t0)

    def __should_abort(self):
        return rospy.is_shutdown() or self.__stop_requested

    def __hold(self, seconds):
        """Wait on the ROS clock; stays responsive to shutdown/stop."""
        end = self.__now() + seconds
        while self.__now() < end and not self.__should_abort():
            time.sleep(0.02)

    def __callback_nav(self, data):
        try:
            p = data.pose.pose.position
            o = data.pose.pose.orientation
            self.__nav = ([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])
            self.__nav_time = self.__now()
        except Exception:
            self.__nav = None

    def __pose_error(self, target):
        """(position error m, attitude error deg) from a fresh navigation sample, else None."""
        nav = self.__nav
        if not nav or self.__now() - self.__nav_time > 3.0:
            return None
        try:
            return (dist(nav[0], target['position']),
                    math.degrees(quat_angle(nav[1], target['quaternion'])))
        except Exception:
            return None

    def __log_pose_error(self, cp):
        """Diagnostics only: how far from the target are we right before reporting?"""
        nav = self.__nav
        if not nav:
            return
        try:
            rospy.loginfo('pose error at checkpoint %d: pos=%.3f m  att=%.1f deg',
                          cp['checkpoint_id'], dist(nav[0], cp['position']),
                          math.degrees(quat_angle(nav[1], cp['quaternion'])))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # mission notifications
    # ------------------------------------------------------------------
    def __notify_mission_start(self):
        rospy.wait_for_service('/event/mission_start')
        res = self.__mission_start()
        rospy.loginfo('mission_start result: success=%s message=%s', res.success, res.message)
        return res.success

    def __notify_mission_finished(self):
        rospy.wait_for_service('/event/mission_finished')
        res = self.__mission_finished()
        rospy.loginfo('mission_finished result: success=%s message=%s (%.1fs left)',
                      res.success, res.message, self.__time_left())
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
                rospy.loginfo('checkpoint_passed result (id=%d): %s', checkpoint_id, result)
                return bool(result.success)
            if self.__time_left() <= HARD_FLOOR:
                break
        self.__checkpoint_client.cancel_goal()
        rospy.logwarn('checkpoint_passed action cancelled')
        return False

    # ------------------------------------------------------------------
    # checkpoint parsing
    # ------------------------------------------------------------------
    @staticmethod
    def __get_checkpoint_id(cp):
        for key in ('checkpoint_id', 'checkpointid', 'id', 'checkpointId'):
            if key in cp:
                try:
                    return int(cp[key])
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def __is_goal(cp):
        for key in ('is_goal', 'goal', 'isGoal'):
            if cp.get(key) is True:
                return True
        return 'goal' in str(cp.get('type', '')).lower() or 'goal' in str(cp.get('name', '')).lower()

    def __load_checkpoints(self):
        """Returns list of dicts: checkpoint_id, position[3], quaternion[4], is_goal."""
        rospy.wait_for_service('/mission/get_checkpoints')
        res = self.__checkpoints()
        rospy.loginfo('get_checkpoints result: success=%s message=%s', res.success, res.message)
        if not res.success:
            return []
        try:
            payload = json.loads(res.message)
        except Exception as e:
            rospy.logwarn('Failed to parse checkpoints payload: %s', e)
            return []
        goal_id = None
        if isinstance(payload, list):
            raw = payload
        else:
            raw = payload.get('checkpoints', payload.get('Checkpoints', []))
            for key in ('goalpoint_id', 'goal_id', 'goalPointId'):
                if payload.get(key) is not None:
                    try:
                        goal_id = int(payload[key])
                    except (TypeError, ValueError):
                        pass
                    break
        rospy.loginfo('goal checkpoint id from payload: %s', goal_id)

        result = []
        for cp in raw:
            if not isinstance(cp, dict):
                continue
            cid = self.__get_checkpoint_id(cp)
            pos = find_position(cp)
            if cid is None or pos is None:
                rospy.logwarn('skipping unparsable checkpoint entry: %s', cp)
                continue
            quat = find_quaternion(cp)
            if quat is None:
                rospy.logwarn('checkpoint %d has NO attitude in payload -> identity (marker may not be centred!)', cid)
                quat = [0.0, 0.0, 0.0, 1.0]
            result.append({
                'checkpoint_id': cid,
                'position': pos,
                'quaternion': quat,
                'is_goal': (goal_id is not None and cid == goal_id) or self.__is_goal(cp),
            })
            rospy.loginfo('parsed checkpoint %d pos=%s quat=%s goal=%s', cid, pos, quat, result[-1]['is_goal'])
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
            t_start, t_close = self.__now(), None
            while not self.__should_abort():
                if self.__command.wait_for_result(rospy.Duration(0.2)):
                    result = self.__command.get_result()
                    rospy.loginfo('move command result: %s', result)
                    if t_close is not None:
                        rospy.loginfo('convergence: within 5cm/4deg after %.1fs, controller finished after %.1fs',
                                      t_close - t_start, self.__now() - t_start)
                    return result.type == CtlCommandResult.TERMINATE_SUCCESS
                if t_close is None:
                    err = self.__pose_error(target)
                    if err and err[0] < 0.05 and err[1] < 4.0:
                        t_close = self.__now()
                if self.__time_left() <= HARD_FLOOR:
                    break
            self.__command.cancel_goal()
            return False
        finally:
            self.__action_active = False

    def __move_to(self, target):
        rospy.loginfo('moving to checkpoint_id=%d pos=%s quat=%s (%.1fs left)',
                      target['checkpoint_id'], target['position'], target['quaternion'], self.__time_left())
        for attempt in range(1, MOVE_RETRIES + 1):
            if self.__should_abort() or self.__time_left() <= HARD_FLOOR:
                return False
            if self.__move_once(target):
                return True
            rospy.logwarn('move to checkpoint %d failed (attempt %d/%d)',
                          target['checkpoint_id'], attempt, MOVE_RETRIES)
        return False

    def __visit(self, cp):
        """Move -> settle -> report. Repeats the cycle if the report is rejected."""
        for attempt in range(1, NOTIFY_RETRIES + 1):
            if self.__should_abort() or self.__time_left() <= SAFETY_MARGIN / 2:
                return False
            if not self.__move_to(cp):
                continue
            self.__hold(HOLD_SECONDS)
            err = self.__pose_error(cp)
            if (err and (err[0] > CORRECT_POS or err[1] > CORRECT_ATT)
                    and self.__time_left() > SAFETY_MARGIN + 15.0):
                rospy.logwarn('checkpoint %d: pose error %.3f m / %.1f deg before report -> corrective move',
                              cp['checkpoint_id'], err[0], err[1])
                if self.__move_to(cp):
                    self.__hold(HOLD_SECONDS)
            self.__log_pose_error(cp)
            if self.__notify_checkpoint_passed(cp['checkpoint_id']):
                rospy.loginfo('checkpoint %d PASSED (attempt %d)', cp['checkpoint_id'], attempt)
                return True
            rospy.logwarn('checkpoint %d report rejected (attempt %d/%d)',
                          cp['checkpoint_id'], attempt, NOTIFY_RETRIES)
        return False

    # ------------------------------------------------------------------
    # main flow
    # ------------------------------------------------------------------
    def __run_user_flow(self, msg):
        try:
            if not self.__notify_mission_start():
                return
            self.__wall_clock = rospy.get_time() <= 0.0
            self.__t0 = self.__now()
            rospy.loginfo('clock source: %s, t0=%.2f', 'wall' if self.__wall_clock else 'ros/sim', self.__t0)

            # USER EDIT AREA: define route planning and game-specific flow here.
            # ***********************************************
            cps = self.__load_checkpoints()
            goal_cps = [c for c in cps if c['is_goal']]
            goal_cp = goal_cps[0] if goal_cps else None
            remaining = [c for c in cps if not c['is_goal']]

            if goal_cp:
                goal_node = node_of(goal_cp)
            elif GOAL_POSE:
                goal_node = (list(GOAL_POSE[:3]), normalize_quat(list(GOAL_POSE[3:7])))
            else:
                goal_node = None

            cur = (list(DOCK_POSITION), list(DOCK_QUATERNION))
            scale = 1.0       # real leg time / predicted leg time (learned, cumulative)
            sum_pred, sum_real = 0.0, 0.0

            while remaining and not self.__should_abort():
                tl = self.__time_left()
                budget = (tl - SAFETY_MARGIN) / (scale * PLAN_PAD)
                route, total = select_route(cur, remaining, goal_node, budget)
                if len(route) < len(remaining) and len(remaining) <= ENDGAME_COUNT:
                    # Skipping anything forfeits the goal bonus anyway, so accept a thinner margin
                    # if finishing EVERYTHING is still predicted to fit.
                    r2, t2 = select_route(cur, remaining, goal_node, (tl - FINISH_MARGIN) / scale)
                    if len(r2) == len(remaining):
                        route, total = r2, t2
                        rospy.loginfo('end-game: finishing all %d remaining checkpoints fits (%.0fs vs %.0fs left)',
                                      len(remaining), t2 * scale, tl)
                dropped = [c['checkpoint_id'] for c in remaining if c not in route]
                rospy.loginfo('plan [%s] (%.0fs left, scale %.2f): visit %s then goal; predicted %.0fs%s',
                              ROUTE_MODE, self.__time_left(), scale, [c['checkpoint_id'] for c in route], total * scale,
                              (' | NOT AFFORDABLE: %s' % dropped) if dropped else '')
                if not route:
                    rospy.logwarn('no further checkpoint fits in the remaining time -> heading to the goal')
                    break
                cp = route[0]
                pred = leg_seconds(cur, node_of(cp))
                t_leg = self.__now()
                ok = self.__visit(cp)
                dt = self.__now() - t_leg
                rospy.loginfo('leg to checkpoint %d took %.1f sim-s (predicted %.1f, ok=%s)',
                              cp['checkpoint_id'], dt, pred * scale, ok)
                if ok and pred > 5.0:
                    sum_pred += pred
                    sum_real += dt
                    scale = min(1.6, max(0.85, sum_real / sum_pred))
                remaining.remove(cp)
                cur = node_of(cp)

            # finish at the goal (report must be inside the time limit)
            if not self.__should_abort() and self.__time_left() > HARD_FLOOR:
                if goal_cp:
                    self.__visit(goal_cp)
                elif GOAL_POSE:
                    self.__move_to({'checkpoint_id': -1,
                                    'position': goal_node[0], 'quaternion': goal_node[1]})
            # ***********************************************

            if not rospy.is_shutdown():
                self.__notify_mission_finished()
        except Exception as exc:
            rospy.logerr('user flow failed: %s', exc)
            try:
                if not rospy.is_shutdown():
                    self.__notify_mission_finished()
            except Exception:
                pass

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
