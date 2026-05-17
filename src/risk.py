# TODO: Implement deterministic engagement-weighted risk scoring here.
#
# Raw engagement formula from AGENTS.MD:
# likes + reposts*2 + comments*1.5 + replies*1.5 + upvotes + helpful_votes + reactions
#
# Keep this calculation in risk.py rather than on the Pydantic Engagement schema,
# so models.py stays focused on validating data shape.
