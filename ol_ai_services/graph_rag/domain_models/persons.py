from typing import Literal

# Define entity and relation schemas
entities = Literal[
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "PRODUCT",
    "EVENT"]

relations = Literal[
    "SUPPLIER_OF",
    "COMPETITOR",
    "PARTNERSHIP",
    "ACQUISITION",
    "WORKS_AT",
    "SUBSIDIARY",
    "BOARD_MEMBER",
    "CEO",
    "PROVIDES",
    "HAS_EVENT",
    "IN_LOCATION",
]

# Define which entities can have which relations
validation_schema = {
    "Person"      : [
        "WORKS_AT",
        "BOARD_MEMBER",
        "CEO",
        "HAS_EVENT"],
    "Organization": [
        "SUPPLIER_OF",
        "COMPETITOR",
        "PARTNERSHIP",
        "ACQUISITION",
        "WORKS_AT",
        "SUBSIDIARY",
        "BOARD_MEMBER",
        "CEO",
        "PROVIDES",
        "HAS_EVENT",
        "IN_LOCATION",
        ],
    "Product"     : ["PROVIDES"],
    "Event"       : ["HAS_EVENT", "IN_LOCATION"],
    "Location"    : ["HAPPENED_AT", "IN_LOCATION"],
    }