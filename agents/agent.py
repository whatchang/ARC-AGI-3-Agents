import json
import logging
import os
import time
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

import requests
import requests.cookies
from pydantic import ValidationError
from requests import Response
from requests.cookies import RequestsCookieJar

from .recorder import Recorder
from .structs import FrameData, GameAction, GameState, Scorecard
from .tracing import trace_agent_session

logger = logging.getLogger()


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm.

    Enforces rate limits across all agents to respect API RPM limits.
    Default: 600 RPM = 10 requests per second (with safety margin of 8 RPS).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, requests_per_second: float = 0):
        """Initialize rate limiter.

        Set to 0 to disable rate limiting (rely on 429 retry instead).
        With batch_size=2, GPU max ~2.9 FPS × 2 = ~5.8 RPS < 10 RPS limit.
        """
        if self._initialized:
            return
        self._initialized = True

        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.enabled = requests_per_second > 0
        self.last_request_time = 0.0
        self._request_lock = threading.Lock()

    def acquire(self):
        """Wait until a request can be made within rate limits."""
        if not self.enabled:
            return  # Rate limiting disabled

        with self._request_lock:
            now = time.time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limited_request(session: requests.Session, method: str, url: str,
                         max_retries: int = 5, **kwargs) -> Response:
    """Make a rate-limited request with exponential backoff retry on 429 errors.

    Args:
        session: requests.Session to use
        method: HTTP method ('get' or 'post')
        url: Request URL
        max_retries: Maximum number of retries on 429 error
        **kwargs: Additional arguments to pass to request

    Returns:
        Response object
    """
    base_delay = 1.0  # Start with 1 second delay

    for attempt in range(max_retries + 1):
        # Wait for rate limiter
        _rate_limiter.acquire()

        # Make request
        if method.lower() == 'get':
            response = session.get(url, **kwargs)
        else:
            response = session.post(url, **kwargs)

        # Check for rate limit error
        if response.status_code == 429:
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + (time.time() % 1)
                logger.warning(f"Rate limit hit (429), retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Rate limit exceeded after {max_retries} retries")

        return response

    return response  # Return last response even if rate limited


class Agent(ABC):
    """Interface for an agent that plays one ARC-AGI-3 game."""

    MAX_ACTIONS: int = 80  # to avoid looping forever if agent doesnt exit
    ROOT_URL: str

    action_counter: int = 0

    timer: float = 0
    agent_name: str
    card_id: str
    game_id: str
    guid: str
    frames: list[FrameData]

    recorder: Recorder
    headers: dict[str, str]
    _session: requests.Session

    # AgentOps tracing attributes
    trace: Any = None
    tags: list[str]

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        tags: Optional[list[str]] = None,
        cookies: requests.cookies.RequestsCookieJar = RequestsCookieJar(),
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name
        self.tags = tags or []
        self.frames = [FrameData(score=0)]
        self._cleanup = True
        if record:
            self.start_recording()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        # Reuse session
        self._session = requests.Session()
        self._session.cookies = deepcopy(cookies)
        self._session.headers.update(self.headers)

    @trace_agent_session
    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(self.frames, self.frames[-1])
            result = self.take_action(action)

            # Handle rate limit - don't count action, don't append frame
            if result is False:
                logger.info(f"{self.game_id} - Rate limited, skipping action (not counted)")
                continue  # Don't increment action_counter

            if result is not None:
                self.append_frame(result)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, score {result.score}, avg fps {self.fps})"
                )
            self.action_counter += 1

        self.cleanup()

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def score(self) -> int:
        return self.frames[-1].score

    @property
    def seconds(self) -> float:
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        elapsed_time = max(self.seconds, 0.1)
        return round(self.action_counter / elapsed_time, 2)

    @property
    def is_playback(self) -> bool:
        return type(self) is Playback

    @property
    def name(self) -> str:
        n = self.__class__.__name__.lower()
        return f"{self.game_id}.{n}"

    def start_recording(self) -> None:
        filename = self.agent_name if self.is_playback else None
        # Support subdirectory structure for custom agents (e.g., "agent_name/seed")
        subdir = getattr(self, 'recording_subdir', None)
        self.recorder = Recorder(prefix=self.name, filename=filename, subdir=subdir)
        logger.info(
            f"created new recording for {self.name} into {self.recorder.filename}"
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    def do_action_request(self, action: GameAction) -> Response:
        data = action.action_data.model_dump()
        if action == GameAction.RESET:
            data["card_id"] = self.card_id
        if self.guid:
            data["guid"] = self.guid
        if action.reasoning:
            data["reasoning"] = action.reasoning
        if self.game_id:
            data["game_id"] = self.game_id

        json_str = json.dumps(data)
        r = rate_limited_request(
            self._session,
            'post',
            f"{self.ROOT_URL}/api/cmd/{action.name}",
            json=json.loads(json_str),
            headers=self.headers,
        )
        try:
            response_json = r.json()
            if "error" in response_json:
                logger.warning(f"Exception during action request: {response_json}")
        except ValueError:
            logger.warning(f"Failed to parse response: {r.status_code} - {r.text}")
        return r

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Submits the specific action and gets the next frame.

        Returns:
            FrameData if successful
            None if validation error
            False if rate limited (to distinguish from None)
        """
        response = self.do_action_request(action)

        # Check if rate limited after all retries
        if response.status_code == 429:
            logger.warning(f"Action skipped due to rate limit - not counting this action")
            return False  # Special marker for rate limit

        try:
            frame_data = response.json()
        except ValueError:
            logger.warning(f"Failed to parse response JSON: {response.status_code}")
            return None

        try:
            frame = FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning(f"Incoming frame data did not validate: {e}")
            return None
        return frame

    def get_scorecard(self) -> Scorecard:
        """Get the scorecard for this agent's game as a Scorecard pydantic object."""
        r = rate_limited_request(
            self._session,
            'get',
            f"{self.ROOT_URL}/api/scorecard/{self.card_id}/{self.game_id}",
            timeout=10,
            headers=self.headers,
        )
        response_data = r.json()
        if "error" in response_data:
            logger.warning(f"Exception during scorecard request: {response_data}")
        return Scorecard.model_validate(response_data)

    def cleanup(self, scorecard: Optional[Scorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent
            if hasattr(self, "recorder") and not self.is_playback:
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                else:
                    scorecard_obj = self.get_scorecard()
                    self.recorder.record(scorecard_obj.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"Exiting: agent reached MAX_ACTIONS of {self.MAX_ACTIONS}, took {self.seconds} seconds ({self.fps} average fps)"
                )
            else:
                logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )
            if hasattr(self, "_session"):
                self._session.close()

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        raise NotImplementedError


class Playback(Agent):
    """An agent that plays back from a recorded session from another agent."""

    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 5

    recorded_actions: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = Recorder(
            prefix=Recorder.get_prefix(self.agent_name),
            guid=Recorder.get_guid(self.agent_name),
        )
        self.recorded_actions = []
        if self.agent_name in Recorder.list():
            try:
                self.recorded_actions = self.filter_actions()
                logger.info(
                    f"Loaded {len(self.recorded_actions)} actions from {self.agent_name}"
                )
            except Exception as e:
                logger.error(f"Failed to load recording {self.agent_name}: {e}")
                self.recorded_actions = []
        else:
            logger.warning(
                f"Recording {self.agent_name} not found in available recordings"
            )

    def filter_actions(self) -> list[dict[str, Any]]:
        return [
            a
            for a in self.recorder.get()
            if "data" in a and "action_input" in a["data"]
        ]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(self.action_counter >= len(self.recorded_actions))

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        loop_start_time = time.time()

        if self.action_counter >= len(self.recorded_actions):
            logger.warning(
                f"No more recorded actions available (counter: {self.action_counter}, total: {len(self.recorded_actions)})"
            )
            return GameAction.RESET

        recorded_data = self.recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]

        action = GameAction.from_id(action_input["id"])
        data = action_input["data"].copy()
        data["game_id"] = self.game_id
        action.set_data(data)
        if "reasoning" in action_input and action_input["reasoning"] is not None:
            action.reasoning = action_input["reasoning"]

        logger.debug(
            f"Playback action {self.action_counter}: {action.name} with data {data}"
        )

        target_frame_time = 1.0 / getattr(self, "PLAYBACK_FPS", 5)
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return action

    def append_frame(self, frame: FrameData) -> None:
        # overwrite append_frame to not double record
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
