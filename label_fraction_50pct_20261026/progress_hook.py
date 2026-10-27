from experiment_common import *
from mmengine.hooks import Hook

class ProgressHook(Hook):
 def after_train_epoch(self,runner):
  write_json(Path(runner.work_dir)/'progress.json',dict(state='training',epoch=runner.epoch+1,epochs=runner.max_epochs,time=__import__('time').time(),pid=os.getpid()))
