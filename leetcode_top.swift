//MARK: 102. 二叉树的层序遍历
func levelOrder(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else {
        return []
    }
    var res = [[Int]]()
    var queue: [TreeNode] = [root] //因为上面做了guard守护
    while !queue.isEmpty {
        var level = [Int]()
        for _ in queue {  //只要遍历当前队列，不需要i
            var node = queue.removeFirst() //出队，用数组模拟
            level.append(node.val)
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
        res.append(level)
    }
    return res
}
