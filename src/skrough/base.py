import attr

import skrough.typing as rgh_typing


@attr.s
class Reduct:
    attrs: rgh_typing.AttrsType = attr.ib()


@attr.s
class Bireduct:
    objects: rgh_typing.ObjectsType = attr.ib()
    attrs: rgh_typing.AttrsType = attr.ib()
