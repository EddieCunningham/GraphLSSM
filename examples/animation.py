import subprocess
import os
from GenModels.GM.States.GraphicalMessagePassing.GraphForwardBackward import *
from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import GraphMessagePasserFBS

class AnimatedGraphMessagePasserFBS( GraphMessagePasserFBS ):

    def messagePassing( self, uWork, vWork, horizontal=False, gif_delay=50, **kwargs ):
        # Wrap uWork and vWork with a drawing tool

        # Path to ../../../tmp
        tmp_folder = '/'.join( os.path.dirname( os.path.realpath( __file__ ) ).split( '/' )[ :-3 ] ) + '/tmp'

        u_style = dict( fontcolor='black',
                        style='bold',
                        color='blue' )
        v_style = dict( fontcolor='black',
                        style='bold',
                        color='green' )
        edge_style = dict( fixedsize='true',
                           color='green' )
        fbs_node_style = dict( fontcolor='black',
                               style='filled',
                               color='blue' )
        styles = { 0: u_style, 1: v_style, 2: edge_style, 3: fbs_node_style }

        i = 0
        u_list_global = None

        def uWorkWrapper( is_base_case, u_list, **kwargs ):
            nonlocal u_list_global
            u_list_global = u_list
            uWork( is_base_case, u_list, **kwargs )

        def vWorkWrapper( is_base_case, v_list, **kwargs ):
            nonlocal i, u_list_global

            # Update the style for the fbs nodes
            node_to_style_key = dict( [ ( n, 3 ) for n in self.fbs ] )

            # Update the style for the u list nodes
            node_to_style_key.update( dict( [ ( int( self.partialGraphIndexToFullGraphIndex( n ) ), 0 ) for n in u_list_global ] ) )

            # Update the style for the v list nodes
            node_to_style_key.update( dict( [ ( int( self.partialGraphIndexToFullGraphIndex( n ) ), 1 ) for n in  v_list[ 0 ] ] ) )

            # Update the style for the v list edges
            edge_to_style_key = dict( [ ( ( int( self.partialGraphIndexToFullGraphIndex( n ) ), e ), 2 ) for n, e in zip( *v_list ) if e is not None ] )

            # Draw the graph
            self.toGraph().advancedDraw( styles=styles, horizontal=horizontal, node_to_style_key=node_to_style_key, edge_to_style_key=edge_to_style_key, output_folder=tmp_folder, output_name='graph_%d'%( i ) )
            i += 1

            vWork( is_base_case, v_list, **kwargs )

        # Run the message passing algorithm
        super().messagePassing( uWorkWrapper, vWorkWrapper, **kwargs )

        # Convert the images to a gif using ImageMagick
        image_paths = [ tmp_folder + '/graph_%d.png'%( j ) for j in range( i ) ]
        commands = [ 'convert', '-loop', '0', '-delay', str( gif_delay ) ] + image_paths + [ 'out.gif' ]
        subprocess.call( commands )

        # Delete the contents of tmp
        subprocess.call( [ 'rm' ] + image_paths )
