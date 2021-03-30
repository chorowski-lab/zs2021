import os
from tqdm import tqdm
import pickle
from sklearn.cluster import MiniBatchKMeans


def kmeans(dataset, args):
    n_batches = len(dataset) // args.bsz
    clustering = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=0, batch_size=args.bsz)
    dataset.set_epoch_ids()
    for i in tqdm(range(n_batches), desc='Batch', ascii=True):
        batch = dataset.get_batch(i, n_batches, args)
        clustering.partial_fit(batch.reshape(-1, batch.shape[-1]))
    return clustering

def train_clustering(dataset, args):
    print('Training clustering {} ...\n'.format(args.clustering))
    if args.clustering == 'kmeans':
        clustering = kmeans(dataset, args)
    elif args.clustering == '':
        clustering = ''
    
    if not os.path.exists(args.clustering_dir):
        os.makedirs(args.clustering_dir)
    pickle.dump(clustering, open(os.path.join(args.clustering_dir, args.clustering), 'wb'))
    return clustering

def load_clustering(args):
    print('Loading trained clustering {} ...\n'.format(args.clustering))
    return pickle.load(open(os.path.join(args.clustering_dir, args.clustering), 'rb'))