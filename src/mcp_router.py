import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from .mcp_server import tools, handle_list_tools, handle_call_tool

router = APIRouter()


@router.post("/mcp")
async def mcp_endpoint(body: dict):
    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params", {})

    try:
        if method == "tools/list":
            tool_list = await handle_list_tools()
            result = {"tools": [t.model_dump() for t in tool_list]}
        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            content_list = await handle_call_tool(name, arguments)
            result = {
                "content": [
                    {"type": c.type, "text": c.text}
                    for c in content_list
                ]
            }
        else:
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                },
                status_code=200,
            )
    except Exception as e:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            },
            status_code=200,
        )

    return JSONResponse(
        content={"jsonrpc": "2.0", "id": req_id, "result": result},
        status_code=200,
    )


@router.options("/mcp")
async def mcp_options():
    return Response(status_code=204)
