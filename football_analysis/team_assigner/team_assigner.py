from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}  # {player_id: whether team 1 or team 2 }

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # to understand this code see the notebook in development_and_analysis folder

        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]

        # get cluster model
        kmeans = self.get_clustering_model(top_half_image)
        # Get the cluster labels forr each pixel
        labels = kmeans.labels_
        # reshape labels to image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])
        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    
    def assign_team_color(self, frame, player_detections):
        # This function takes one frame and corresponding player tracks and assigns team color to each player in frame and **finds team colors**
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"] 
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Now we are going to divide player_colors into 2 colors, one is white and another one is green
        ## note that player_colors are float values and all are not same(because centers of the image color).
        ## so using kmeans we again divide that into 2 teams
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame,player_bbox,player_id):
        # This function assigns players to team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id +=1 # convert 0->1 and 1->2
        self.player_team_dict[player_id] = team_id

        return team_id