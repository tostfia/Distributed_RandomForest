resource "aws_ecs_cluster" "forest_cluster" {
  name = var.cluster_name

  tags = {
    Project = var.project_name
  }
}
