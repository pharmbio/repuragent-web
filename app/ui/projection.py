'''One function that turns state into component updates.

Every handler that can change the workspace returns `render(state)`, so `layout.py`
names a single `workspace_outputs` list instead of a dozen hand-maintained ones that
drift apart.
'''

from __future__ import annotations

from typing import Any, Optional, Tuple

import gradio as gr

from app.state import UIState
from app.ui.approval import approval_updates
from app.ui.conversation_panel import conversation_panel_update
from app.ui.progress_panel import progress_update


def control_updates(state: UIState):
    '''Send/Stop interactivity.

    Send is disabled while this thread is running: a second submit does not queue
    visibly, it blocks on the per-thread lock and then fires minutes later against a
    conversation that has moved on. Stop is only offered when there is something to
    stop.

    Parameters:
    ---------
    state (UIState): the state to derive interactivity from.

    Returns:
    ----------
    updates (tuple): Send and Stop interactivity. Send is disabled while the thread runs, because a second submit blocks on the per-thread lock and then fires minutes later against a conversation that has moved on.
    '''

    running = state.is_running
    return gr.update(interactive=not running), gr.update(interactive=running)


def render(
    state: UIState,
    *,
    clear_input: bool = False,
    live: bool = False,
) -> Tuple[Any, ...]:
    '''The workspace output tuple every handler returns.

    `clear_input` is true only on the yield that accepts a submission. Later yields
    leave the textbox alone, so text typed while the agents work is not wiped out
    from under the user — which is what a blanket `gr.update(value="")` on every
    yield did.

    `live` marks a frame emitted purely to advance streamed text. Those arrive
    several times a second and cannot have changed the file list or the plan, so the
    side panels are skipped: re-sending them swaps their DOM, which made the file
    list impossible to scroll during a run.

    Parameters:
    ---------
    state (UIState): the state to project.
    clear_input (boolean): empty the textbox — true only on the yield that accepts a submission, since a blanket clear wipes text typed while the agents work.
    live (boolean): a token frame, which skips both side panels.

    Returns:
    ----------
    outputs (tuple): the workspace output tuple every handler returns.
    '''

    banner, approve_button, changes_button = approval_updates(state.pending_approval)
    send_button, stop_button = control_updates(state)
    return (
        state,
        list(state.messages),
        gr.update(value="") if clear_input else gr.skip(),
        gr.skip() if live else conversation_panel_update(state),
        banner,
        approve_button,
        changes_button,
        send_button,
        stop_button,
        # Read straight from plan.md, so the panel cannot disagree with the record.
        gr.skip() if live else progress_update(state),
    )


def auth_status_text(state: UIState) -> str:
    base = (
        "**Sign in to use Repuragent.**"
        if not state.is_authenticated
        else f"**Signed in as** `{state.user_email}`"
    )
    if state.auth_error:
        return f"{base}\n\n{state.auth_error}"
    return base


def auth_message(text: str, success: bool = True) -> str:
    return f"{'**Success:**' if success else '**Error:**'} {text}"


def render_auth(state: Optional[UIState], *, clear_input: bool = False) -> Tuple[Any, ...]:
    '''Workspace outputs plus the sign-in controls.

    Parameters:
    ---------
    state (UIState): the state to project, or None before sign-in.
    clear_input (boolean): empty the textbox on this yield.

    Returns:
    ----------
    outputs (tuple): the workspace outputs plus the sign-in controls.
    '''

    state = state or UIState()
    signed_in = bool(state.is_authenticated)
    return (
        *render(state, clear_input=clear_input),
        auth_status_text(state),
        gr.update(visible=signed_in),  # logout button
        gr.update(visible=not signed_in),  # login button
        gr.update(interactive=signed_in and state.is_verified),  # new task button
    )


__all__ = ["auth_message", "auth_status_text", "control_updates", "render", "render_auth"]
