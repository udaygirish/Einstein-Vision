# Einstein-Vision
RBE 595 Project- Einstein Vision Implementation


### Base environment activation for simulation 

source ~/Projects/WPI/computer_vision/project3/blender-4.0.2-linux-x64/blender_env/bin/activate


### Popular Self Driving Car Dataset - OpenLane

#### Format

    {
        "intrinsic":                            <float> [3, 3] -- camera intrinsic matrix
        "extrinsic":                            <float> [4, 4] -- camera extrinsic matrix
        "lane_lines": [                         (k lanes in `lane_lines` list)
            {
                "category":                     <int> -- lane category
                                                            0: 'unkown',
                                                            1: 'white-dash',
                                                            2: 'white-solid',
                                                            3: 'double-white-dash',
                                                            4: 'double-white-solid',
                                                            5: 'white-ldash-rsolid',
                                                            6: 'white-lsolid-rdash',
                                                            7: 'yellow-dash',
                                                            8: 'yellow-solid',
                                                            9: 'double-yellow-dash',
                                                            10: 'double-yellow-solid',
                                                            11: 'yellow-ldash-rsolid',
                                                            12: 'yellow-lsolid-rdash',
                                                            20: 'left-curbside',
                                                            21: 'right-curbside'
                "visibility":                   <float> [n, ] -- visibility of each point
                "uv":[                          <float> [2, n] -- 2d lane points under image coordinate
                    [u1,u2,u3...],
                    [v1,v2,v3...]
                ],
                "xyz":[                         <float> [3, n] -- 3d lane points under camera coordinate
                    [x1,x2,x3...],
                    [y1,y2,y3...],
                    [z1,z2,z3...],

                ],
                "attribute":                    <int> -- left-right attribute of the lane
                                                            1: left-left
                                                            2: left
                                                            3: right
                                                            4: right-right
                "track_id":                     <int> -- lane tracking id
            },
            ...
        ],
        "file_path":                            <str> -- image path
    }
