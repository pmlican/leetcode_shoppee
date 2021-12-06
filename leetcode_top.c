
// shopee高频leetcode
//MARK: leetcode 215. 数组中的第K个最大元素 (快排提前返回)
void swap(int* a, int* b) {
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int* nums, int l, int r) {
    int i = l;
    int x = nums[r];  //取右边作为基准值
    // i 和 j指针 划分4个区域
    // 比x小  |i 比x大   |j 未查看   |r
    for (int j = l; j < r; j++) {
        //如果比
        if (nums[j] <= x) {
            //如果小于基准值x，移到左边
            swap(&nums[i], &nums[j]);
            i++;
        }
    }
    // 最后把基本值，放到i的位置，就是中间的位置
    swap(&nums[i], &nums[r]);
    return i;
}

// 随机一个partition , 防止退化成O(n^2)
int randomPartition(int *nums, int l, int r) {
    // random一个随机数，在 i - r之间
    //比如[3,2,1,|5,6,4], 随机5和4直接的值， 3(元素个数)(0-3之间) + 3(5的下标)
    //r - l + 1 = 个数， 然后加上 l, 就是大于l
    int i = rand() % (r - l + 1) + l;
    swap(&nums[i], &nums[r]);
    return partition(nums, l, r);
}

int quickSelect(int *nums, int l, int r, int index) {
    int q = randomPartition(nums, l, r);
    if (q == index) {
        return nums[q];
    } else {
        //如果 q < index， 如果index比q大，那就在右区间，递归右区间
        return q < index ? quickSelect(nums, q + 1, r, index) : quickSelect(nums, l, q - 1, index);
    }
}

int findKthLargest(int* nums, int numsSize, int k){
    srand(time(0)); //生成一个时间种子，因为随机数是根据当前时间戳生成的
    return quickSelect(nums, 0, numsSize - 1, numsSize - k);
}

//MARK: 88. 合并两个有序数组 (逆向双指针)

void merge(int* nums1, int nums1Size, int m, int* nums2, int nums2Size, int n){

}




//MARK: 94. 二叉树的中序遍历
// 递归解法
void inorder(struct TreeNode* root, int* returnSize, int* res) {
    if (root == NULL) {
        return;
    }
    inorder(root->left, returnSize, res);
    res[(*returnSize)++] = root->val;
    inorder(root->right, returnSize, res);
}

int* inorderTraversal(struct TreeNode* root, int* returnSize){
    int* res = malloc(sizeof(int) * 201);
    *returnSize = 0;
    inorder(root, returnSize, res);
    return res;
}

// 用栈来模拟递归，先左，中，右
int* inorderTraversal(struct TreeNode* root, int* returnSize) {
    *returnSize = 0;
    int* res = malloc(sizeof(int) * 501);
    struct TreeNode** stk = malloc(sizeof(struct TreeNode*) * 501);
    int top = 0;
    while (root != NULL || top > 0) {
        while (root != NULL) {
            stk[top++] = root;
            root = root->left;
        }
        root = stk[--top];
        res[(*returnSize)++] = root->val;
        root = root->right;
    }
    return res;
}

//MARK: 912. 排序数组
void swap(int *x,int *y) {
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

int partition(int* a, int l, int r) {
    int x = a[r], i = l;
    for (int j = l; j < r; j++) {
        if (a[j] <= x) {
            swap(&a[i++], &a[j]);
        }
    }
    swap(&a[i], &a[r]);
    return i;
}

int randomPartition(int* a, int l, int r) {
    //random(min: l, max: r) 类似这样随机一个l到r之间的数
    //比如[3,2,1,|5,6,4], 随机5和4直接的值， 3(元素个数)(0-3之间) + 3(5的下标)
    int i = rand() % (r - l + 1) + l;
    swap(&a[i], &a[r]);
    return partition(a, l, r);
}

void quickSort(int *nums, int l, int r) {
    if (l < r) {
        int p = randomPartition(nums, l, r); //解决本身为升序数组，降为O(n^2)
        quickSort(nums, l, p - 1);
        quickSort(nums, p + 1, r);
    }
}
int* sortArray(int* nums, int numsSize, int* returnSize){
    srand(time(0));
    quickSort(nums, 0, numsSize - 1);
    *returnSize = numsSize;
    return nums;
}
