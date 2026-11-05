# Lesson notes
Basic Agent patter - ReAct : React + Action
This follows Thought->Action->Observation->Final Answer

There are other agent patterns as well-
Agent pattern	How it works	Example
ReAct	Think → Act → Observe	Research assistant
Tool-calling agent	LLM selects and calls tools	Weather assistant
Planning agent	Creates a plan, then executes it	Travel planner
Multi-agent system	Multiple agents collaborate	Research and writing team
Workflow-based agent	Follows predefined steps and decisions	Loan application processing

How do we orchestrate the agent's steps? - Using LangGraph
LangGraph allows you to build cyclic workflows, manage state, call tools, add conditional routing, and implement human approval.
It can implement ReAct agents, but it can also implement many other architectures.

Interview takeaway: An LLM provides reasoning and decision-making, while tools allow an agent to interact with external systems and perform actions.

Suppose a user asks:

"Find the average transaction amount from my company's database."
User asks for average transaction amount.
LLM identifies that database access is required.
LLM generates a database-query action.
Python executes the database tool.
Database returns transaction records or an aggregated result.
LLM presents the result to the user.


Learning about groups-
action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to selection action
match = action_re.match("Action: average_dog_weight: Border Collie")

print(match.group(0))
print(match.group(1))
print(match.group(2))
print(match.groups())

group(0) → entire matched text, including Action:
group(1) → first parenthesized expression
group(2) → second parenthesized expression
groups() → tuple containing groups 1 and 2, not group 0

actions[0].groups() returns the values captured by parentheses () in the regex pattern, not just any parentheses in the original text.
action_re = re.compile(r'^Action: (\w+): (.*)$')
It has two capturing groups:
- (\w+) → Group 1
- (.*) → Group 2
