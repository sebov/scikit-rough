import attr

AttrsType = list[int]
ObjectsType = list[int]


@attr.s
class Reduct:
    attrs: AttrsType = attr.ib()


@attr.s
class Bireduct:
    objects: ObjectsType = attr.ib()
    attrs: AttrsType = attr.ib()
