"""Node package root.

Every module placed anywhere under this package is auto-imported by
registry.discover(). Group nodes into subpackages by domain (image/,
conditioning/, sampling/, utils/ ...). To add a node: drop a file, tag the
class with @register. No edits anywhere else.
"""
