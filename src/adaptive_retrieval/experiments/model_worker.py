import multiprocessing as mp

def _serve(connection, manifest, device, chunks, with_reranker):
    try:
        from .prepare import load_encoder
        from .backends import CrossEncoderReranker

        encoder = load_encoder(manifest["model_path"], device)

        def encode(question):
            text = manifest["query_instruction"] + question
            if len(encoder.tokenizer.encode(text)) > encoder.max_seq_length:
                raise ValueError("Query exceeds embedding token limit")
            return encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

        encode("Warm up enterprise retrieval.")
        reranker = (CrossEncoderReranker(manifest["reranker_path"], device) if with_reranker else None)
        if reranker:
            reranker.rerank("Warm up", [(0, 1.0)], chunks)
        connection.send((True, "ready"))
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            try:
                result = (encode(value) if command == "encode" else reranker.rerank(value[0], value[1], chunks))
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except EOFError:
        pass
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class ModelWorker:
    def __init__(self, manifest, device, chunks, with_reranker=True):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, manifest, device, chunks, with_reranker), daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(240):
            raise TimeoutError("No response from model worker in 240s")
        ok, result = self.connection.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def encode(self, question):
        self.connection.send(("encode", question))
        return self._receive()

    def rerank(self, question, ranking, chunks):
        self.connection.send(("rerank", (question, ranking)))
        return self._receive()

    def close(self):
        # Release the worker anyway if when startup or a pipe operation failed
        try:
            if self.process.is_alive():
                try:
                    self.connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join()
        finally:
            self.connection.close()
