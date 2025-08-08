from google.adk.agents.llm_agent import Agent
# from google.adk.tools.mcp_tool.mcp_toolset import(
#     MCPToolset, SseServerParams
# )
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams


# async def get_tools_async():
#     """Gets tools from the File System MCP Server."""
#     tools, exit_stack = await MCPToolset.from_server(
#         connection_params=SseServerParams(
#             url="http://localhost:8080/officer/mcp",
#         )
#     )
#     print("MCP Toolset created successfully.")
#     return tools, exit_stack


root_agent = Agent(

    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='A mcp cleint agent that can interact with the MCP server. SPeically to create , deal with officers',
    tools=[
            MCPToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url="http://localhost:8080/officer/mcp",
                )
            )
        ],
)
