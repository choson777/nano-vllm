from typing import Optional 


class AsyncLLM:
    """A Large Language Model (LLM) for online inference."""

    def __init__(
        self,
        model: str,
        enforce_eager:,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        **
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = ModelConfig(
            model,
            tokenizer,
            trust_remote_code=trust_remote_code,
            seed=seed,
            use_dummy_weights=use_dummy_weights,
        )
        self.parallel_config = ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.cache_config = CacheConfig(
            block_size, max_num_blocks_per_req, gpu_memory_utilization, swap_space
        )
        self.sched_config = SchedConfig(
            sched_policy,
            max_batch_size,
            max_tokens_per_batch,
            model_name=model,
            profiling_file=profiling_file,
            parallel_config=self.parallel_config,
            proactive_offloading=proactive_offloading,
            num_min_free_blocks_threshold=num_min_free_blocks_threshold,
            num_queues_for_prediction=num_queues_for_prediction,
            use_skip_join=use_skip_join,
        )
        self.llm_engine = LLMEngine(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self.sched_config,
        )
        # request_id => event
        self.request_events = {}
        # request_id => step_output
        self.step_outputs = {}
        self.is_engine_running = False
        self.kicking_request_id: Optional[str] = None
        self.timeout_interval = 1  # seconds

    async def engine_step(self, kicking_request_id: Optional[str] = None):
        """Kick the engine to process the waiting requests."""
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        # Yield to the event loop to allow other coroutines to run
        # while is_engine_running is True. This let the engine to add new
        # requests into the queue.
        await asyncio.sleep(0)
        step_outputs, _ = self.llm_engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None

        # Notify the waiting coroutines that there are new outputs ready.
        for step_output in step_outputs:
            request_id = step_output.request_id
            # The request may get aborted
            if request_id in self.request_events:
                self.request_events[request_id].set()
                self.step_outputs[request_id] = step_output

    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> Request:
        """Generate outputs for a single request.

        This method is a coroutine. It adds the request into the waiting queue,
        kicks the LLMEngine for generation, and streams the outputs to the caller.

        Args:
            request_id: The unique id of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            sampling_params: The sampling parameters of the request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        if prompt is None and prompt_token_ids is None:
            raise ValueError("prompt or prompt_token_ids must be provided")

        arrival_time = time.time()
        # Create an event to notify us when there is new output from the LLMEngine.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        # Add the request to the LLMEngine.
        self.llm_engine.add_request(
            prompt, prompt_token_ids, sampling_params, arrival_time, request_id
        )

        # Keep kicking the engine to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

            # Kick the engine if the engine is not running.
            if not self.is_engine_running:
                try:
                    for _ in range(self.parallel_config.pipeline_parallel_size):
                        await self.engine_step(request_id)
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e

            # Wait for new output. The group_event will be set in engine_step
            # when there is new output available for the request.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(
                    request_event.wait(), timeout=self.timeout_interval
                )
            except asyncio.TimeoutError:
                # logger.info("timeout")
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            step_output = self.step_outputs[request_id]
            yield step_output

            # Once finished, release the resources of the request.
            if step_output.is_finished:
                logger.info(f"Finished request {request_id}.")

                del self.step_outputs[request_id]
                del self.request_events[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        logger.info(f"Aborted request {request_id}.")

        self.llm_engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.step_outputs:
            del self.step_outputs[request_id]

        # To prevent deadlock when a request is aborted while the engine is
        # running.
        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.kicking_request_id = None
