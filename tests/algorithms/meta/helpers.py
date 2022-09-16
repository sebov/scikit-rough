from skrough.algorithms.meta.describe import DescriptionNode

DUMMY_NODE_NAME = "dummy"
LEAF_VALUE = "value"

DUMMY_NODE = DescriptionNode(
    node_name=DUMMY_NODE_NAME,
    name="name",
    short_description="short_description",
    long_description="long_description",
)


def get_describe_dict(describe_node: DescriptionNode):
    """Prepare helper dict structure for better navigation in tests."""
    result = {}
    if describe_node.children is not None:
        for element in describe_node.children:
            result[element.node_name] = get_describe_dict(element)
    else:
        result[LEAF_VALUE] = describe_node
    return result
