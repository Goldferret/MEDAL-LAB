#!/usr/bin/python

from madsci.client.workcell_client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

workcell_client = WorkcellClient(
  workcell_manager_url=os.getenv("WORKCELL_MANAGER_URL", "http://localhost:8005")
)
wf_def = WorkflowDefinition(
  name="Test Workflow",
  steps=[
    StepDefinition(
      name="Test Step 0",
      node="DOFBOT_Pro_1", # Must exist in workcell nodes
      action="transfer",
      args={
        "locations": [90, 90],
        "movement_keys": ["angled_above", "angled_above", "horizontal"] # Apporach, Return, and Grab methods (in order)
      }
    )
  ]
)

result = workcell_client.submit_workflow(workflow=wf_def,)
# Alternatively, specify the workflow as a path
#result = workcell_client.submit_workflow(workflow="path/to/test.workflow.yaml")

# You can also not await the workflow results, and query later
#result = workcell_client.submit_workflow(workflow=wf_def, await_completion=False)
#time.sleep(10)
result = workcell_client.query_workflow(result.workflow_id)
