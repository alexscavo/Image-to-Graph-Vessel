import numpy as np
import torch
from ignite.engine import Engine, Events

from monai.handlers import TensorBoardHandler
from monai.visualize.img2tensorboard import plot_2d_or_3d_image


class TensorBoardImageHandlerWithTag(TensorBoardHandler):
    def __init__(
        self,
        summary_writer=None,
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform=lambda x: x,
        output_transform=lambda x: x,
        global_iter_transform=lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        frame_dim: int = -3,
        max_frames: int = 64,
        output_tag: str = "output",
        input0_tag: str = "input_0",
        input1_tag: str = "input_1",
    ) -> None:
        super().__init__(summary_writer=summary_writer, log_dir=log_dir)
        self.interval = interval
        self.epoch_level = epoch_level
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.frame_dim = frame_dim
        self.max_frames = max_frames
        self.max_channels = max_channels
        self.output_tag = output_tag
        self.input0_tag = input0_tag
        self.input1_tag = input1_tag

    def attach(self, engine: Engine) -> None:
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine) -> None:
        step = self.global_iter_transform(engine.state.epoch if self.epoch_level else engine.state.iteration)

        # ----- images -----
        show_images = self.batch_transform(engine.state.batch)[0][self.index]
        if isinstance(show_images, torch.Tensor):
            show_images = show_images.detach().cpu().numpy()
        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output)[0] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_images).__name__}."
                )
            plot_2d_or_3d_image(
                data=show_images[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
                tag=self.input0_tag,
            )

        # ----- labels -----
        show_labels = self.batch_transform(engine.state.batch)[1][self.index]
        if isinstance(show_labels, torch.Tensor):
            show_labels = show_labels.detach().cpu().numpy()
        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise TypeError(
                    "batch_transform(engine.state.batch)[1] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_labels).__name__}."
                )
            plot_2d_or_3d_image(
                data=show_labels[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
                tag=self.input1_tag,
            )

        # ----- outputs -----
        show_outputs = self.output_transform(engine.state.output)[self.index]
        if isinstance(show_outputs, torch.Tensor):
            show_outputs = show_outputs.detach().cpu().numpy()
        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output) must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_outputs).__name__}."
                )
            plot_2d_or_3d_image(
                data=show_outputs[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
                tag=self.output_tag,  # <-- key change
            )

        self._writer.flush()
