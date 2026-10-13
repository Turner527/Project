"""MCP-сервер с инструментами анализа KPI.

Запуск (stdio):  python -m tools.mcp_server
Сервер ничего не считает сам: регистрирует функции из tools.core, их схемы строятся по аннотациям.
Внимание: по stdio идёт протокол, поэтому в этом процессе нельзя печатать в stdout.
"""
from mcp.server.mcpserver import MCPServer

from tools.core import TOOLS

INSTRUCTIONS = (
    "Инструменты аналитика стратегического менеджмента банка. Все числа в ответах посчитаны кодом по витрине "
    "kpi.xlsx; каждый ответ содержит call_id. Для вопросов «почему отстаём» достаточно diagnose_deviation."
)


def build_server() -> MCPServer:
    server = MCPServer("bank-kpi-analyst", instructions=INSTRUCTIONS)
    for fn in TOOLS.values():
        server.add_tool(fn)
    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
