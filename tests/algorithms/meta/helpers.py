from skrough.structs.description_node import DescriptionNode

LEAF_VALUE = "value"

DUMMY_NODE = DescriptionNode(
    name="name",
    short_description="short_description",
    long_description="long_description",
)


def get_describe_dict(describe_node: DescriptionNode):
    """Prepare helper dict structure for better navigation in tests.

    DN := DescriptionNode
    DN.children[DN(node_name="0"), DN(node_name="1"), DN(node_name="2")]
        =>
    {
        "0": {"value": "0"},
        "1": ...
    }
    """
    result = {}
    if describe_node.children is not None:
        for element in describe_node.children:
            result[element.node_name] = get_describe_dict(element)
    else:
        result[LEAF_VALUE] = describe_node
    return result
