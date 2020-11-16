import itertools

import more_itertools
from attrdict import AttrDict
import util


class JointInfo:
    def __init__(self, joints, edges=()):
        if isinstance(joints, dict):
            self.ids = joints
        elif isinstance(joints, (list, tuple)):
            self.ids = JointInfo.make_id_map(joints)
        elif isinstance(joints, str):
            self.ids = JointInfo.make_id_map(joints.split(','))
        else:
            raise Exception

        self.names = list(sorted(self.ids.keys(), key=self.ids.get))
        self.n_joints = len(self.ids)

        if isinstance(edges, str):
            self.stick_figure_edges = []
            for path_str in edges.split(','):
                joint_names = path_str.split('-')
                for joint_name1, joint_name2 in more_itertools.pairwise(joint_names):
                    if joint_name1 in self.ids and joint_name2 in self.ids:
                        edge = (self.ids[joint_name1], self.ids[joint_name2])
                        self.stick_figure_edges.append(edge)
        else:
            self.stick_figure_edges = edges

        # the index of the joint on the opposite side (e.g. maps index of left wrist to index
        # of right wrist)
        self.mirror_mapping = [
            self.ids[JointInfo.other_side_joint_name(name)] for name in self.names]

    def update_names(self, new_names):
        if isinstance(new_names, str):
            new_names = new_names.split(',')

        self.names = new_names
        new_ids = AttrDict()
        for i, new_name in enumerate(new_names):
            new_ids[new_name] = i
        self.ids = new_ids

    @staticmethod
    def make_id_map(names):
        return AttrDict(dict(zip(names, itertools.count())))

    @staticmethod
    def other_side_joint_name(name):
        if name.startswith('l'):
            return 'r' + name[1:]
        elif name.startswith('r'):
            return 'l' + name[1:]
        else:
            return name

    def select_joints(self, selected_joint_ids):
        new_names = [self.names[i] for i in selected_joint_ids]
        new_edges = [(selected_joint_ids.index(i), selected_joint_ids.index(j))
                     for i, j in self.stick_figure_edges
                     if i in selected_joint_ids and j in selected_joint_ids]
        return JointInfo(new_names, new_edges)
