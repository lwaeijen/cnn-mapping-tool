from ..dsesegment import DSESegment
from itertools import permutations, product, islice
from ...utils import chunks, pareto_cost_config, CNNParetoSet
from math import ceil, log, floor
from ...model.configs import SegmentConfig, LayerConfig, NetworkConfig, NetworkConfigs
from copy import deepcopy
from ...optimizers import foldBuffers
import multiprocessing
from copy import deepcopy
import operator
from tqdm import tqdm
import signal
import logging

class NaiveSegment(DSESegment):

    def __init__(self, **kwargs):

        #Init super
        super(NaiveSegment, self).__init__(**kwargs)

    def __config_space(self,
            notiling=False,  #if set to true tiles will be set to the featuremap dimensions
            **kwargs
        ):

        #Get output layer
        out = self.last

        #determine loop levels
        inner_loops=['x_i','y_i']+( ['zi_i'] if out['type'] in ['convolution'] else [] )+['zo_i']
        outer_loops=['x_o','y_o']+( ['zi_o'] if out['type'] in ['convolution'] else [] )+['zo_o']

        #define the design search space for this segment (naive schedule, tiles are full size)
        space={
            'Tx':               [out['xo']],
            'Ty':               [out['yo']],
            'Tzo':              [out['zo']],
            'order':            [inner_loops+outer_loops],
        }
        if out['type'] in ['convolution']:
            space['Tzi'] = [out['zi']/out['groups']]
        else:
            space['Tzi'] = [1]

        #options for the output store level
        space['output_store_at']=range(len(inner_loops)+len(outer_loops))

        #For each convolution layer there is a store data and store weights level
        invalid_correction_factor=1.0
        store_keys=[]
        for l in self.nodes():
            if self.nodes[l]['type'] in ['convolution']:
                space[str(l)+'.store_at']          = range(5) # if not self.TESTING else range(1)
                space[str(l)+'.store_at_weights']  = range(5) # if not self.TESTING else range(1)
                store_keys+=[str(l)+'.store_at', str(l)+'.store_at_weights']

                #all options where the weights are above the data are invalid, so we correct for these invalid schedules to get the correct number of schedules
                lvls=5
                invalid_correction_factor*= float(lvls+1)/float(lvls*2)

        num_configs=int(reduce(operator.mul, map(len, space.values()))*invalid_correction_factor)

        return space, store_keys, num_configs

    #Generate all possible configurations
    def __all_configs_raw(self,
            **kwargs
        ):

        #get the space
        space, store_keys, num_configs=self.__config_space(**kwargs)

        #yield all configurations
        for cfg_v in product(*space.values()):
            #combine keys with values
            cfg=dict( (k,v) for k,v in zip(space.keys(), cfg_v) )

            #It is not possible to compute the weights after they are needed to compute the data,
            # hence we reject all these configurations
            valid=True
            for data, weights in chunks(store_keys,2):
                if cfg[weights] < cfg[data]:
                    valid=False
                    break
            if not valid:
                continue

            #resolve store levels
            for key in store_keys+['output_store_at']:
                cfg[key]=cfg['order'][cfg[key]]

            #set default compute levels
            for l in self.nodes():
                if self.nodes[l]['type'] in ['convolution']:
                    cfg[l+'.compute_at']=cfg[l+'.store_at']
                    cfg[l+'.compute_at_weights']=cfg[l+'.store_at_weights']

            yield cfg

    def fix_raw_config(self,
            cfg,
            buffer_folding=True,
            force_fold=True,
            **kwargs
        ):
        #translate dict into new-style segment config
        layer_configs=[]
        for l in self.nodes():
            if self.nodes[l]['type'] not in ['convolution']:
                #still need to add a layerconfig for this layer, but init with none
                layer_configs += [LayerConfig(l, None, None, None, None)]
                continue

            #create convolutional layer config
            layer_configs+= [LayerConfig(
                l,                          #name
                cfg[l+'.store_at'],         #store level
                cfg[l+'.store_at_weights'],
                cfg[l+'.compute_at'],
                cfg[l+'.compute_at_weights']
            )]

        config=SegmentConfig(
            #Segment name
            self.name,

            #order
            cfg['order'],

            #tiling
            {
                'Tx'  : cfg['Tx'],
                'Ty'  : cfg['Ty'],
                'Tzo' : cfg['Tzo'],
                'Tzi' : cfg['Tzi'],
            },

            #output store level
            cfg['output_store_at'],


            #layer configs
            layer_configs
        )

        #possibly apply some optimizations
        if buffer_folding:
            foldBuffers(self.net, config, force=force_fold)

        #return segment config
        return config

    def all_configs(self,
            **kwargs
        ):
        #split into raw config, and then adding proper model configs
        #done to speed up multiprocessing
        for cfg in self.__all_configs_raw(**kwargs):
            yield self.fix_raw_config(cfg, **kwargs)

    def explore(self, pareto=True, unique=True, **kwargs):
        if pareto:
            return self.__explore_pareto(**kwargs)
        elif unique:
            return self.__explore_unique(**kwargs)
        else:
            return self.__explore_all(**kwargs)

    def __explore_all(self, **kwargs):
        for cfg in self.all_configs(**kwargs):
            #yield evaluated configuration
            yield self.eval_config(cfg, **kwargs)

    def __explore_unique(self, **kwargs):
        unique=set()
        for cfg in self.all_configs(**kwargs):
            #eval configuration
            self.eval_config(cfg, **kwargs)

            #get costs
            cost=cfg.cost_tuple

            #only yield unique costs
            l=len(unique)
            unique.add(cost)
            if len(unique)>l:
                yield cfg


    #Bruteforce all configurations and return pareto points
    def __explore_pareto(self, **kwargs):

        class Worker(multiprocessing.Process):
            def __init__(self, task_q, result_q, kwargs, evaluator):
                super(Worker, self).__init__()
                self.task_q=task_q
                self.result_q = result_q
                self.kwargs=kwargs

                #process private pareto set
                self.points = CNNParetoSet()

                #eval_config function is not threadsafe, so we copy that state here for use in the workers
                self.evaluator = deepcopy(evaluator)

            def run(self):

                #set worker to ignore keyboard interupts
                signal.signal(signal.SIGINT, signal.SIG_IGN)

                #keep getting tasks from the queue
                while True:

                    #get task from the queue
                    task = self.task_q.get()

                    #are we done yet?
                    if task is None:
                        break

                    #valid task, extract
                    for cfg_raw in task:

                        #generate proper models
                        cfg=self.evaluator.fix_raw_config(cfg_raw, **self.kwargs)

                        #update costs for this configurations
                        self.evaluator.eval_config(cfg, **self.kwargs)

                        #get costs
                        cost=cfg.cost_tuple

                        #add point to pareto set
                        point=cost + (cfg,)
                        self.points+=point

                    #signal queue we are done
                    self.task_q.task_done()

                #end of tasks

                #send results to merger thread
                self.result_q.put(list(self.points))

                #guarantee memory cleanup
                del self.evaluator
                del self.points


        class ResultMerger(multiprocessing.Process):
            def __init__(self, task_q, result_q):
                super(ResultMerger, self).__init__()
                self.points=CNNParetoSet()
                self.task_q=task_q
                self.result_q=result_q

            def run(self):
                #set worker to ignore keyboard interupts
                signal.signal(signal.SIGINT, signal.SIG_IGN)

                while True:
                    task = self.task_q.get()
                    if task==None:
                        #assign the local points to the parent class
                        #also strip the cost of, no longer required for sorting
                        self.result_q.put([cost_cfg[-1] for cost_cfg in self.points])
                        break

                    for point in task:
                        self.points+=point

        #first select only points with unique costs
        self.points=None

        #job queue
        num_procs=multiprocessing.cpu_count() #make sure there are enough threads to keep the cores occupied
        task_q = multiprocessing.JoinableQueue(maxsize=num_procs*2) #and enough places in the work queue to keep the threads busy
        result_q = multiprocessing.Queue(maxsize=num_procs*2)
        merged_result_q = multiprocessing.Queue(maxsize=1)

        #start separate thread to merge partial pareto sets of worker threads
        merger=ResultMerger(result_q, merged_result_q)
        merger.start()

        #start worker threads
        workers = [ Worker(task_q, result_q, kwargs,  self) for _ in xrange(num_procs) ]
        for w in workers:
            w.start()

        #put configurations in the work queue
        #init iterator
        try:
            cfg_it = self.__all_configs_raw(**kwargs)
            slice_size=50000
            _,_,num_configs=self.__config_space(**kwargs)

            #init a progress bar if required by log level
            pbar = tqdm(total=num_configs, unit=" configs") if logging.getLogger().getEffectiveLevel()<=logging.INFO else None

            #keep adding tasks to the queue
            while True:
                cfgs=list(islice(cfg_it, slice_size ))
                if not cfgs:
                    break
                task_q.put(cfgs)
                if pbar: pbar.update(len(cfgs))

            #close the progress bar if any
            if pbar: pbar.close()

            # block until all workers are done processing the jobs
            task_q.join()

        except KeyboardInterrupt:
            #kill al workers and merger
            for w in workers:
                w.terminate()
                w.join()
            merger.terminate()
            merger.join()
            #reraise the keyboard interrupt
            raise

        # stop all the workers gracefully
        for i in range(len(workers)):
            task_q.put(None)
        for w in workers:
            w.join()
            del w

        #signal merger thread to stop
        result_q.put(None)
        merger.join()
        del merger

        #get results
        self.configs=merged_result_q.get()

        return self.configs
