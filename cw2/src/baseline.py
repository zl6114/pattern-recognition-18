# Utilities
import json
from   tqdm import tqdm
from   os   import path

# Calculation
import numpy    as np 
from   scipy.io import loadmat

# ML
from sklearn.neighbors import NearestNeighbors

def loadData():
    '''
        Loads file containing features
    '''
    
    bp = path.dirname( __file__ )
    fp = path.abspath( path.join( bp, '../PR_data/feature_data.json' ) )
    with open( fp, 'r' ) as file:
        features = json.load( file )

    return np.asarray( features )


def loadMat():
    '''
        Loads matlab file 
    '''

    bp = path.dirname( __file__ )
    fp = path.abspath( path.join( bp, '../PR_data/cuhk03_new_protocol_config_labeled.mat' ) )

    return loadmat( fp )


def createAugmentedSets( mat, data ):
    '''
        Creates Augmented train/query/gallery arrays which contain:
        features x 2048 | camId x 1 | label x 1 

        Returns train, query, gallery augmented matrices
    '''

    labels = mat[ 'labels' ].flatten() # Load labels
    camIds = mat[ 'camId' ].flatten() # Load camId

    # Load indexes
    train_idxs   = mat[ 'train_idx' ].flatten()
    query_idxs   = mat[ 'query_idx' ].flatten()
    gallery_idxs = mat[ 'gallery_idx' ].flatten()


    # Create Train Set
    train_set, train_label, train_camId   = [], [], []

    for i in train_idxs:
        train_set.append( data[ i - 1 ] )
        train_label.append( labels[ i - 1 ] )
        train_camId.append( camIds[ i - 1] )

    train_set   = np.asarray( train_set )
    train_label = np.asarray( train_label )
    train_camId = np.asarray( train_camId )

    train_augmented = np.vstack( ( train_set.T, train_camId, train_label ) ).T

    # Create Query Set
    query_set, query_label, query_camId = [], [], []

    for i in query_idxs:
        query_set.append( data[ i - 1] )
        query_label.append( labels[ i - 1 ] )
        query_camId.append( camIds[ i - 1 ] )

    query_set   = np.asarray( query_set )
    query_label = np.asarray( query_label )
    query_camId = np.asarray( query_camId )

    query_augmented = np.vstack( ( query_set.T, query_camId, query_label ) ).T

    # Create Gallery Set
    gallery_set, gallery_label, gallery_camId = [], [], [] 

    for i in gallery_idxs:
        gallery_set.append( data[ i - 1] )
        gallery_label.append( labels[ i - 1 ] )
        gallery_camId.append( camIds[ i - 1 ] )

    gallery_set   = np.asarray( gallery_set )
    gallery_label = np.asarray( gallery_label )
    gallery_camId = np.asarray( gallery_camId )

    gallery_augmented = np.vstack( ( gallery_set.T, gallery_camId, gallery_label ) ).T

    return train_augmented, query_augmented, gallery_augmented

def baseline( train_augmented, query_augmented, gallery_augmented, n_neighbors = 20, metric = 'euclidean' ):
    '''
        Returns rank list for n_neighbors
    '''
    query_rank_list = []

    for i in tqdm( range( query_augmented.shape[ 0 ] ) ):

        query_label = query_augmented[ i, -1 ].astype( int )

        # Remove rows which have the same camId AND label
        gallery_reduced = gallery_augmented[ ~np.logical_and( ( gallery_augmented[ :, -1 ] == query_augmented[ i ][ -1 ] ),
                                                            ( gallery_augmented[ :, -2 ] == query_augmented[ i ][ -2 ] )
                                        ) ]
        
        # Train KNN
        X = gallery_reduced[ :, : - 2 ] # All rows, but in each row, remove last 2 columns ( camId and label )
        Y = gallery_reduced[ :, - 1 ] # All rows in the last column ( the labels )
        
        KNN = NearestNeighbors( n_neighbors = n_neighbors, metric = metric )
        KNN.fit( X, Y )    
        
        # Test query point
        X_test = query_augmented[ i ][ : -2 ].reshape( 1, -1 ) # Remove last 2 columns ( camId and label )
        
        distances, indices = KNN.kneighbors( X_test ) # Neighbours are ordered closest to furthest
        
        # Compare
        distances = distances.flatten()
        indices   = indices.flatten()

        rank_list = [ gallery_reduced[ ind, -1 ].astype( int ) == query_label for ind in indices ]
        query_rank_list.append( rank_list )

    return np.asarray( query_rank_list )


def main():

    data = loadData()
    mat  = loadMat()

    t, q, g = createAugmentedSets( mat, data )

    qrl = baseline( t, q, g )

    print( 'ass')

    useless = 5 + 10

if __name__ == '__main__':
    main()
